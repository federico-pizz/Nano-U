"""Read-only questions about a built model: counts, summary, serialisation."""

import io
import os
import tempfile

import tf_keras as keras

Model = keras.Model


def count_parameters(model: Model) -> int:
    """Return total trainable parameter count."""
    return int(model.count_params())


def get_model_summary(model: Model) -> str:
    """Return model summary as string (captures model.summary())."""
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    return buf.getvalue()


def get_model_config(model: Model) -> dict:
    """Return dict with model_name, input_shape, parameter_count, layer_count."""
    input_shape = list(model.input_shape[1:]) if model.input_shape else []
    return {
        "model_name": model.name,
        "input_shape": input_shape,
        "parameter_count": count_parameters(model),
        "layer_count": len(model.layers),
    }


def validate_model_serialization(model: Model) -> bool:
    """Save and load model; return True if successful."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        model.save(path, save_format="h5")
        loaded = keras.models.load_model(path)
        return loaded is not None and loaded.count_params() == model.count_params()
    except Exception:
        return False
    finally:
        if os.path.exists(path):
            os.unlink(path)
