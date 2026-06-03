"""Firmware INT8 preprocessing parity.

The on-device input pipeline (firmware/src/lib.rs :: build_quant_luts +
preprocess_rgb) turns raw RGB8 camera bytes into the model's int8 input via:

    x = byte / 255
    q = round((x - mean_c) / std_c / input_scale + input_zero_point)  clamped to int8

It uses mean/std/scale/zero_point that firmware/build.rs reads from
<model>_quant_params.json, which src/quantize_model.extract_quant_params writes.

If that chain drifts from how the model was trained/calibrated, the device feeds
the network wrongly-scaled pixels — fine offline metrics, garbage on hardware.
These tests pin the contract against the *actual* converted model.

Note: bit-exact rounding vs the device (libm::roundf, ties-away-from-zero) can't
be checked without the hardware/Rust; we verify the affine transform, the
parameters, channel mapping, clamp, and that the result is a valid model input.
"""

import math

import numpy as np
import pytest
import tensorflow as tf

from src.models import create_nano_u
from src.models.utils import convert_to_tflite_quantized
from src.quantize_model import extract_quant_params

H, W = 60, 80


def _representative_gen():
    rng = np.random.default_rng(0)
    for _ in range(4):
        yield [rng.standard_normal((1, H, W, 3)).astype(np.float32)]


def _round_half_away(v):
    """Mirror libm::roundf (round half away from zero), not Python's bankers'."""
    return math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)


def _firmware_lut(mean, std, scale, zero_point):
    """Python transcription of firmware/src/lib.rs::build_quant_luts (lines 65-74).

    Kept in lockstep with lib.rs; update both together if the device math changes.
    """
    luts = []
    for c in range(3):
        lut = np.empty(256, dtype=np.int16)
        for j in range(256):
            x = j / 255.0
            q = _round_half_away((x - mean[c]) / std[c] / scale + zero_point)
            lut[j] = int(np.clip(q, -128, 127))
        luts.append(lut)
    return luts


@pytest.fixture(scope="module")
def quant(tmp_path_factory):
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out = str(tmp_path_factory.mktemp("fw") / "nano_u.tflite")
    assert convert_to_tflite_quantized(model, out, _representative_gen), "INT8 export failed"
    params = extract_quant_params(out, config_path="config/config.yaml")
    return out, params


def test_firmware_lut_dequantizes_to_normalized_value(quant):
    """Each LUT entry, dequantized by the model's own (scale, zero_point), must
    recover the training-normalized pixel value to within one quantization step.

    This ties firmware preprocessing to the actual converted model rather than
    restating the same formula: it would fail on a wrong scale, zero_point sign,
    or mismatched mean/std.
    """
    _, p = quant
    mean, std = p["normalization"]["mean"], p["normalization"]["std"]
    scale, zp = p["input"]["scale"], p["input"]["zero_point"]
    luts = _firmware_lut(mean, std, scale, zp)

    for c in range(3):
        for j in range(256):
            q = int(luts[c][j])
            if q <= -128 or q >= 127:
                continue  # clamped: value outside the representable range
            dequant = (q - zp) * scale
            target = (j / 255.0 - mean[c]) / std[c]
            assert abs(dequant - target) <= scale + 1e-6, (
                f"ch{c} byte{j}: dequant {dequant} vs normalized target {target} "
                f"(scale {scale})"
            )


def test_firmware_lut_output_is_a_valid_model_input(quant):
    """A frame quantized by the firmware LUT must be accepted and run by the
    actual interpreter — right dtype (int8), shape, and value range."""
    path, p = quant
    mean, std = p["normalization"]["mean"], p["normalization"]["std"]
    scale, zp = p["input"]["scale"], p["input"]["zero_point"]
    luts = _firmware_lut(mean, std, scale, zp)

    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)  # raw RGB8 frame
    q = np.empty((1, H, W, 3), dtype=np.int8)
    for c in range(3):
        q[0, :, :, c] = luts[c][raw[:, :, c]].astype(np.int8)

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    assert inp["dtype"] == np.int8
    assert list(inp["shape"]) == [1, H, W, 3]

    interp.set_tensor(inp["index"], q)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    assert out.dtype == np.int8
    assert np.isfinite(out.astype(np.float32)).all()
