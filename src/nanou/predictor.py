"""One forward interface for both model formats.

An INT8 `.tflite` and an FP32 Keras model are scored by the same caller, so the
quantisation step cannot hide behind a different evaluation path. Which format
gets scored is the caller's choice: it passes the file it wants, and
`load_predictor` picks the class from the suffix. Nothing falls back to another
format behind the caller's back.

Quantised input and output are converted here, so callers always see
probabilities in [0, 1].
"""

import os
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from nanou.architectures import PadToMatch
from nanou.metrics import BinaryIoU

CUSTOM_OBJECTS = {"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Logistic sigmoid in numpy. Shared with the device path, so equal logits give equal probabilities."""
    return 1.0 / (1.0 + np.exp(-logits))


class Predictor:
    """What both formats share: raw output -> probabilities, and a pass over a dataset.

    Subclasses implement `raw`, the model's own output for a batch, and set
    `fmt` ("fp32" or "int8"), which names the report files.
    """

    fmt = ""

    def __init__(self, path: str | None = None):
        self.path = path

    def raw(self, images: np.ndarray) -> np.ndarray:
        """The model's own output for a batch: logits."""
        raise NotImplementedError

    def predict(self, images) -> np.ndarray:
        """Probabilities in [0, 1] for a batch of normalised images.

        Both architectures end in a linear conv, so the output is always
        logits and always gets the sigmoid - as it does on the device. There
        used to be a probe that skipped the sigmoid when the first image's
        output happened to lie in [0, 1]; it made host and device disagree on
        exactly those images, so it is gone.
        """
        return logits_to_probs(self.raw(np.asarray(images, dtype=np.float32)))

    def run(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict over a tf.data.Dataset of (images, masks) batches.

        Returns:
            ``(probs, masks, images)`` as float32 arrays, in dataset order.
        """
        all_probs, all_masks, all_imgs = [], [], []
        for imgs, masks in dataset:
            imgs = imgs.numpy()
            all_probs.append(np.asarray(self.predict(imgs), dtype=np.float32))
            all_masks.append(masks.numpy())
            all_imgs.append(imgs)
        return (
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_masks, axis=0),
            np.concatenate(all_imgs, axis=0),
        )


class KerasPredictor(Predictor):
    """An FP32 Keras model - loaded from ``.h5``/``.keras``, or passed in already built."""

    fmt = "fp32"

    def __init__(self, model_or_path):
        if isinstance(model_or_path, (str, os.PathLike)):
            super().__init__(str(model_or_path))
            # QAT checkpoints carry tfmot wrappers, so they only load inside quantize_scope.
            with tfmot.quantization.keras.quantize_scope():
                with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
                    self.model = keras.models.load_model(self.path, compile=False)
        else:
            super().__init__(None)
            self.model = model_or_path

    def raw(self, images):
        """Run the batch through the model in inference mode."""
        return np.asarray(self.model(images, training=False), dtype=np.float32)


class TFLitePredictor(Predictor):
    """A ``.tflite`` file, run through the TFLite interpreter one image at a time.

    An INT8 model's input is quantised with the model's own scale and zero
    point, and its output dequantised the same way - the arithmetic the
    firmware does on the board.
    """

    def __init__(self, path):
        super().__init__(str(path))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")
            self.interpreter = tf.lite.Interpreter(model_path=self.path)
        self.interpreter.allocate_tensors()

        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]
        self._in_q = (
            self._in["quantization"] if self._in.get("quantization", (0.0, 0)) != (0.0, 0) else None
        )
        self._out_q = (
            self._out["quantization"]
            if self._out.get("quantization", (0.0, 0)) != (0.0, 0)
            else None
        )
        quantized = self._in["dtype"] in (np.int8, np.uint8)
        if quantized and self._in_q is None:
            raise RuntimeError(
                f"{self.path} expects quantized input but carries no quantization params"
            )
        self.fmt = "int8" if quantized else "fp32"

    def raw(self, images):
        """Run each image through the interpreter, (de)quantising at the edges."""
        in_dtype, out_dtype = self._in["dtype"], self._out["dtype"]
        outs = []
        for i in range(images.shape[0]):
            inp = images[i : i + 1]
            if in_dtype in (np.int8, np.uint8):
                scale, zero = self._in_q
                inp = np.clip(
                    np.round(inp / scale) + zero,
                    np.iinfo(in_dtype).min,
                    np.iinfo(in_dtype).max,
                ).astype(in_dtype)
            else:
                inp = inp.astype(in_dtype)
            self.interpreter.set_tensor(self._in["index"], inp)
            self.interpreter.invoke()
            o = self.interpreter.get_tensor(self._out["index"])
            if out_dtype in (np.int8, np.uint8) and self._out_q is not None:
                scale, zero = self._out_q
                o = (o.astype(np.float32) - zero) * scale
            else:
                o = o.astype(np.float32)
            outs.append(o)
        return np.concatenate(outs, axis=0)


def load_predictor(model_or_path) -> Predictor:
    """The predictor for a model: a path ending in ``.tflite`` or anything Keras."""
    if isinstance(model_or_path, (str, os.PathLike)) and str(model_or_path).lower().endswith(
        ".tflite"
    ):
        return TFLitePredictor(model_or_path)
    return KerasPredictor(model_or_path)
