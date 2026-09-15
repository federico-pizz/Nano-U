"""Round-trip test for INT8 TFLite export + quant-param extraction.

Covers the deployment-critical path: a Keras model is converted to a
full-integer INT8 TFLite model (src/models/utils.convert_to_tflite_quantized),
and the quantization parameters that firmware/build.rs depends on are read back
out (src/quantize_model.extract_quant_params).
"""

import os

import numpy as np
import pytest

from nanou.architectures import create_nano_u
from nanou.quantize import convert_to_tflite_quantized, extract_quant_params

H, W = 60, 80


def _representative_gen():
    rng = np.random.default_rng(0)
    for _ in range(2):
        yield [rng.standard_normal((1, H, W, 3)).astype(np.float32)]


def test_int8_export_roundtrip(tmp_path):
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    out_path = str(tmp_path / "nanou.tflite")

    convert_to_tflite_quantized(model, out_path, _representative_gen)
    assert os.path.getsize(out_path) > 0

    params = extract_quant_params(out_path, config_path="config/presets/tinyagri.yaml")

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


def test_missing_calibration_images_raise(tmp_path, monkeypatch):
    """A missing calibration split must fail loudly, not fall back to noise.

    Noise-calibrated INT8 models load and run while being numerically useless,
    so the failure has to surface at export time.
    """
    from nanou import quantize as qm

    cfg = {
        "data": {
            "input_shape": (H, W, 3),
            "paths": {"processed": {"val": {"img": str(tmp_path / "does_not_exist")}}},
        }
    }
    monkeypatch.setattr(qm, "load_config", lambda *_a, **_k: cfg)

    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    keras_path = str(tmp_path / "nanou.h5")
    model.save(keras_path)

    with pytest.raises(FileNotFoundError, match="No calibration images"):
        qm.quantize_model(keras_path, str(tmp_path / "nanou.tflite"))


def test_calibration_split_is_validated(tmp_path, monkeypatch):
    """Only 'val' and 'train' are accepted - never the held-out test split."""
    from nanou import quantize as qm

    cfg = {
        "data": {"input_shape": (H, W, 3), "paths": {"processed": {}}},
        "quantization": {"calibration_split": "test"},
    }
    monkeypatch.setattr(qm, "load_config", lambda *_a, **_k: cfg)

    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    keras_path = str(tmp_path / "nanou.h5")
    model.save(keras_path)

    with pytest.raises(ValueError, match="calibration_split"):
        qm.quantize_model(keras_path, str(tmp_path / "nanou.tflite"))


# -- the paper's central claim: INT8 tracks its float source --------------------


def _learnable_dataset(tmp_path, n=8):
    """A tiny dataset with real structure: bright top half, and that is the mask.

    The shared `tmp_png_dataset` fixture has random masks, which a model can
    only fit by predicting one constant everywhere - and a constant output is
    exactly the case where INT8 and float agree trivially. This one is
    learnable in a couple of epochs, so the comparison below has spatially
    varying probabilities to compare.
    """
    import cv2

    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    imgs, masks = [], []
    for i in range(n):
        img = (rng.random((H, W, 3)) * 40).astype(np.uint8)
        img[: H // 2] = np.clip(img[: H // 2].astype(int) + 190, 0, 255).astype(np.uint8)
        mask = np.zeros((H, W), np.uint8)
        mask[: H // 2] = 255

        img_path = str(img_dir / f"frame_{i:03d}.png")
        mask_path = str(mask_dir / f"frame_{i:03d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        imgs.append(img_path)
        masks.append(mask_path)
    return imgs, masks


def test_int8_output_tracks_float_source(tmp_path):
    """The exported INT8 model must agree with the Keras model it came from.

    This is the claim the whole project rests on - QAD carries accuracy through
    quantization - and it is the assertion that would have made the TinyAgri
    0.700 mIoU collapse impossible to ship. Everything else in this file checks
    that the export *ran*; this checks that it means the same thing.
    """
    import tf_keras as keras

    from nanou.data import make_dataset
    from nanou.metrics import compute_segmentation_metrics
    from nanou.predictor import run_inference
    from nanou.train import set_global_seed

    set_global_seed(0)
    imgs, masks_on_disk = _learnable_dataset(tmp_path / "data")
    # batch_size=1 on purpose: the fit below has to converge every time, and
    # what decides that is the number of optimizer steps, not the number of
    # epochs - see the comment on the fit call.
    dataset = make_dataset(
        imgs, masks_on_disk, batch_size=1, shuffle=False, augment=False, target_size=(H, W)
    )

    # Train on something learnable first: an untrained model, or one fitted to
    # noise, settles on a single constant probability, and at a constant output
    # INT8 and float agree for free.
    model = create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    # 60 epochs of 8 single-sample steps. The outcome is bimodal - this model
    # either learns the split and spans the full [0,1] range, or sits at one
    # constant - and the deciding factor is step count, because BatchNorm's
    # moving statistics are what inference mode reads back. 40 epochs at
    # batch_size=2 (160 steps) failed the spread check below in 3 of 8 seeds on
    # CPU and 6 of 16 on GPU; 480 steps passed 32 of 32 across both, with the
    # worst spread at 0.95. Seeding does not make this deterministic, so the
    # margin has to come from the step count. Costs about 7 seconds.
    model.fit(dataset, epochs=60, verbose=0)

    # Calibrate on the same real images the model is scored on, not on noise.
    def representative_gen():
        for batch, _ in dataset.unbatch().batch(1).take(8):
            yield [batch.numpy().astype(np.float32)]

    tflite_path = str(tmp_path / "nanou.tflite")
    convert_to_tflite_quantized(model, tflite_path, representative_gen)

    float_probs, gt, _ = run_inference(model, dataset)
    int8_probs, int8_gt, _ = run_inference(tflite_path, dataset)

    assert float_probs.shape == int8_probs.shape
    assert np.array_equal(gt, int8_gt)
    # Preconditions, so a degenerate model cannot make this test pass for free:
    # the float model must produce a range of probabilities, and the two paths
    # must not return the same array (`run_inference` falls back to a Keras
    # sibling when a .tflite cannot be allocated).
    assert float_probs.max() - float_probs.min() > 0.5, "the float model output is near-constant"
    assert not np.array_equal(float_probs, int8_probs), "the INT8 path did not actually run"

    mean_abs_delta = float(np.mean(np.abs(float_probs - int8_probs)))
    assert mean_abs_delta < 0.05, f"INT8 probabilities drifted from float by {mean_abs_delta:.4f}"

    float_miou = compute_segmentation_metrics(float_probs, gt)["miou"]
    int8_miou = compute_segmentation_metrics(int8_probs, int8_gt)["miou"]
    assert abs(float_miou - int8_miou) < 0.05, (
        f"mIoU moved from {float_miou:.4f} (float) to {int8_miou:.4f} (int8)"
    )
