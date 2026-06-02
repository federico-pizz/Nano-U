"""Round-trip test for INT8 TFLite export + quant-param extraction.

Covers the deployment-critical path: a Keras model is converted to a
full-integer INT8 TFLite model (src/models/utils.convert_to_tflite_quantized),
and the quantization parameters that firmware/build.rs depends on are read back
out (src/quantize_model.extract_quant_params).
"""

import os

import numpy as np
import pytest

from src.models import create_nano_u
from src.models.utils import convert_to_tflite_quantized
from src.quantize_model import extract_quant_params

H, W = 60, 80


def _representative_gen():
    rng = np.random.default_rng(0)
    for _ in range(2):
        yield [rng.standard_normal((1, H, W, 3)).astype(np.float32)]


def test_int8_export_roundtrip(tmp_path):
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out_path = str(tmp_path / "nano_u.tflite")

    ok = convert_to_tflite_quantized(model, out_path, _representative_gen)
    assert ok, "INT8 conversion failed"
    assert os.path.getsize(out_path) > 0

    params = extract_quant_params(out_path, config_path="config/config.yaml")

    assert params["input"]["dtype"] == "int8"
    assert params["output"]["dtype"] == "int8"
    assert params["input"]["shape"] == [1, H, W, 3]
    assert params["output"]["shape"] == [1, H, W, 1]

    for io in ("input", "output"):
        scale = params[io]["scale"]
        zero = params[io]["zero_point"]
        assert scale is not None and np.isfinite(scale) and scale > 0.0
        assert zero is not None and -128 <= zero <= 127

    # Normalization must be carried through for the firmware build script.
    assert len(params["normalization"]["mean"]) == 3
    assert len(params["normalization"]["std"]) == 3
