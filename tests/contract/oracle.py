"""Measurement helpers shared by the baseline capture and the contract tests.

Nothing here asserts. Each function answers one question about a shipped
artefact - how many parameters, which operators, what metrics on the held-out
split - and both sides of the comparison call the same function, so a captured
baseline and a live measurement are always produced the same way.

The module deliberately avoids importing anything from ``src`` at import time:
the TensorFlow-free helpers (hashing, flatbuffer parsing) stay usable when the
heavy imports are not wanted.
"""

import hashlib
import json
import os

import numpy as np

# Datasets that ship with a model in ``models/`` and a preset in ``config/presets``.
DATASETS = {
    "TinyAgri": "tinyagri",
    "BotanicGarden": "botanicgarden",
}

MODEL_NAME = "nanou"

# The three parameter counts the README quotes.
FLOAT_PARAMS = 4212  # plain float32 architecture
QAT_PARAMS = 4688  # after tfmot wraps it in fake-quant nodes
INT8_PARAMS = 3357  # stored constants in the exported INT8 graph


def project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def artifact_paths(dataset):
    """The shipped Keras, TFLite and quant-params files for one dataset."""
    root = project_root()
    models_dir = os.path.join(root, "models", dataset)
    return {
        "tflite": os.path.join(models_dir, f"{MODEL_NAME}.tflite"),
        "quant_params": os.path.join(models_dir, f"{MODEL_NAME}_quant_params.json"),
        "keras": os.path.join(models_dir, f"{MODEL_NAME}.h5"),
        "summary": os.path.join(root, "results", dataset, "pipeline_summary.json"),
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# -- TFLite introspection, flatbuffer level (no interpreter needed) ------------


def _read_flatbuffer(tflite_path):
    from tensorflow.lite.tools import flatbuffer_utils

    return flatbuffer_utils.read_model(tflite_path)


def tflite_param_counts(tflite_path):
    """Element counts of every constant tensor, split by dtype.

    A constant tensor is one whose buffer carries data - weights, folded
    batch-norm scales, biases. Activations have empty buffers and are excluded.
    The total is the 3,357 figure: 3,063 int8 weights plus 294 int32 biases.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    model = _read_flatbuffer(tflite_path)
    counts = {}
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            buf = model.buffers[tensor.buffer]
            if buf.data is None or len(buf.data) == 0:
                continue
            shape = tensor.shape
            n = int(np.prod(shape)) if shape is not None and len(shape) else 1
            key = type_names.get(tensor.type, str(tensor.type)).lower()
            counts[key] = counts.get(key, 0) + n
    counts["total"] = sum(counts.values())
    return counts


def tflite_op_codes(tflite_path):
    """Sorted operator names used by the graph.

    Reads the operator table rather than asking the interpreter, so a Flex
    (SELECT_TF_OPS) fallback shows up as a custom op instead of silently
    working on the host and failing on the device.
    """
    from tensorflow.lite.python import schema_py_generated as schema

    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    model = _read_flatbuffer(tflite_path)
    out = []
    for code in model.operatorCodes:
        builtin = max(
            getattr(code, "builtinCode", 0) or 0, getattr(code, "deprecatedBuiltinCode", 0) or 0
        )
        if builtin:
            out.append(names.get(builtin, str(builtin)))
        elif code.customCode:
            out.append("CUSTOM:" + code.customCode.decode())
        else:
            out.append(names.get(0, "ADD"))
    return sorted(out)


def tflite_io(tflite_path):
    """Input and output dtype, shape and quantization, via the interpreter."""
    import warnings

    import tensorflow as tf

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    def describe(details):
        scale, zero = details["quantization"]
        return {
            "dtype": np.dtype(details["dtype"]).name,
            "shape": [int(x) for x in details["shape"]],
            "scale": float(scale),
            "zero_point": int(zero),
        }

    return {
        "input": describe(interpreter.get_input_details()[0]),
        "output": describe(interpreter.get_output_details()[0]),
        "size_bytes": os.path.getsize(tflite_path),
    }


# -- Parameter counts, Keras side ---------------------------------------------


def float_param_count(preset="tinyagri", model_name=MODEL_NAME):
    """Parameters of the plain float32 student, built from the shipped preset.

    The builders take no default architecture any more, so the preset is the
    only statement of what the shipped model is - which is the right source for
    a contract test anyway. It is built the way ``train_model`` builds it, from
    the model's resolved config.
    """
    from nanou.architectures import build_from_config
    from nanou.config import resolve

    return int(build_from_config(resolve(preset).for_model(model_name)).count_params())


def keras_param_count(h5_path):
    """Parameters of a saved model, QAT wrappers included."""
    import tensorflow_model_optimization as tfmot
    from tf_keras import models, utils

    from nanou.architectures import PadToMatch
    from nanou.metrics import BinaryIoU

    with tfmot.quantization.keras.quantize_scope():
        with utils.custom_object_scope({"BinaryIoU": BinaryIoU, "PadToMatch": PadToMatch}):
            model = models.load_model(h5_path, compile=False)
    return int(model.count_params())


# -- Metrics on the held-out split --------------------------------------------


def score_split(preset, model_path, split="test", batch_size=8):
    """Score one on-disk model over one split and return the metrics dict.

    This is the same forward path ``nanou.evaluate.evaluate`` uses -
    ``make_dataset`` then ``Predictor.run`` then ``compute_segmentation_metrics``
    - reproduced here without the plotting and report writing, so the oracle
    does not depend on where those artefacts get written.
    """
    from nanou.config import resolve
    from nanou.data import make_dataset, sorted_by_frame
    from nanou.metrics import compute_segmentation_metrics
    from nanou.predictor import load_predictor

    config = resolve(preset)
    split_cfg = config.split_paths[split]
    img_dir, mask_dir = split_cfg["img"], split_cfg["mask"]

    imgs = sorted_by_frame(
        [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith(".png")]
    )
    masks = sorted_by_frame(
        [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f.endswith(".png")]
    )
    if not imgs:
        raise RuntimeError(f"no images for split={split} under {img_dir}")

    norm = config.normalization
    shape = config.input_shape
    dataset = make_dataset(
        imgs,
        masks,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        target_size=(shape[0], shape[1]),
        mean=norm["mean"],
        std=norm["std"],
    )

    probs, gt, _ = load_predictor(model_path).run(dataset)
    return compute_segmentation_metrics(probs, gt)


# -- Comparison helpers --------------------------------------------------------

SCALAR_METRICS = ("precision", "recall", "dice", "miou", "f0.5", "f1", "f2")


def flat_metrics(metrics):
    """The scalar headline metrics, as a plain name -> float dict."""
    return {k: float(metrics[k]) for k in SCALAR_METRICS if k in metrics}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
