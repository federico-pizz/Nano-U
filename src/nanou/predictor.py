"""One forward path for both model formats.

An INT8 `.tflite` and an FP32 Keras model are scored by the same caller, so the
quantisation step cannot hide behind a different evaluation path. Quantised
input and output are converted here, so callers always see probabilities in
[0, 1].
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from nanou.architectures import PadToMatch
from nanou.metrics import BinaryIoU


def run_inference(model_or_path, dataset, custom_objects=None):
    """Run a Keras model or an INT8 TFLite file over a tf.data.Dataset.

    Accepts a filesystem path (``.tflite`` / ``.h5`` / ``.keras``) or an already
    loaded Keras model. Returns ``(probs, masks, imgs)`` as float32 numpy arrays
    with probabilities in [0, 1] (sigmoid applied when the model emits logits).

    This is the single forward path shared by ``evaluate_and_plot`` and the CV
    harness, so device-format (INT8 TFLite) and float models are scored
    identically.
    """
    interpreter = None
    model = None
    input_details = output_details = None
    co = custom_objects or {"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}

    if isinstance(model_or_path, str):
        is_tflite = model_or_path.lower().endswith(".tflite")
        if is_tflite:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")
                interpreter = tf.lite.Interpreter(model_path=model_or_path)
            try:
                interpreter.allocate_tensors()
            except RuntimeError:
                # SELECT_TF_OPS / Flex: fall back to a Keras sibling if present.
                base = os.path.splitext(model_or_path)[0]
                found = next(
                    (base + e for e in (".keras", ".h5", ".hdf5") if os.path.exists(base + e)), None
                )
                if not found:
                    raise
                with tfmot.quantization.keras.quantize_scope():
                    with keras.utils.custom_object_scope(co):
                        model = keras.models.load_model(found, compile=False)
                interpreter = None
        else:
            with tfmot.quantization.keras.quantize_scope():
                with keras.utils.custom_object_scope(co):
                    model = keras.models.load_model(model_or_path, compile=False)
    else:
        model = model_or_path  # already-loaded Keras model

    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        in_det, out_det = input_details[0], output_details[0]
        in_dtype, out_dtype = in_det["dtype"], out_det["dtype"]
        q_scale, q_zero = (None, None)
        if in_det.get("quantization", (0.0, 0)) != (0.0, 0):
            q_scale, q_zero = in_det["quantization"]
        out_qs, out_qz = (None, None)
        if out_det.get("quantization", (0.0, 0)) != (0.0, 0):
            out_qs, out_qz = out_det["quantization"]

    def _forward(batch_imgs):
        """Return raw model output (logits or probs) for a batch as numpy."""
        if interpreter is not None:
            outs = []
            for i in range(batch_imgs.shape[0]):
                inp = batch_imgs[i : i + 1].numpy()
                if in_dtype in (np.int8, np.uint8):
                    if q_scale is None:
                        raise RuntimeError(
                            "TFLite expects quantized input but no quantization params were found"
                        )
                    inp = np.clip(
                        np.round(inp / q_scale) + q_zero,
                        np.iinfo(in_dtype).min,
                        np.iinfo(in_dtype).max,
                    ).astype(in_dtype)
                else:
                    inp = inp.astype(in_dtype)
                interpreter.set_tensor(in_det["index"], inp)
                interpreter.invoke()
                o = interpreter.get_tensor(out_det["index"])
                if out_dtype in (np.int8, np.uint8) and out_qs is not None:
                    o = (o.astype(np.float32) - out_qz) * out_qs
                else:
                    o = o.astype(np.float32)
                outs.append(o)
            return np.concatenate(outs, axis=0)
        return np.asarray(model(batch_imgs, training=False), dtype=np.float32)

    # Probe once whether the model emits probabilities or logits.
    probe_imgs = next(iter(dataset))[0]
    probe = _forward(probe_imgs[0:1])
    is_prob = bool(probe.min() >= -1e-3 and probe.max() <= 1.0 + 1e-3)

    all_probs, all_masks, all_imgs = [], [], []
    for imgs, masks in dataset:
        raw = _forward(imgs)
        probs = raw if is_prob else (1.0 / (1.0 + np.exp(-raw)))
        all_probs.append(np.asarray(probs, dtype=np.float32))
        all_masks.append(masks.numpy())
        all_imgs.append(imgs.numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_imgs, axis=0),
    )
