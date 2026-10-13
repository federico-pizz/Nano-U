"""Quantization-Aware Training (QAT) utilities for Keras models."""

import tensorflow_model_optimization as tfmot
import tf_keras as keras


def apply_qat_to_model(model: keras.Model, strict: bool = True) -> keras.Model:
    """Apply Quantization-Aware Training to a Keras Functional model.

    Uses `tfmot.quantization.keras.quantize_model` directly to ensure that
    layer patterns like Conv2D + BatchNormalization are correctly matched and
    folded during the fake-quantization process, which is critical for
    preserving performance.

    Args:
        model: A compiled or uncompiled Keras Functional model.
        strict: When True (the default) a wrapping failure raises. QAD depends
            on fake-quantization being present from epoch 1, so a silent
            fallback would produce a run that looks like QAD, trains in pure
            float, and only shows the damage after INT8 export. Pass
            strict=False to opt into the legacy fail-soft behavior.

    Returns:
        A QAT-annotated model ready for training.

    Raises:
        RuntimeError: if quantization fails and ``strict`` is True.
    """
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print(f"  QAT applied: {qat_model.count_params():,} params (was {model.count_params():,})")
        return qat_model

    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"QAT wrapping failed for model '{model.name}': {exc}. "
                "Training would silently fall back to float and the INT8 export "
                "would lose accuracy. Fix the model or set qat_enabled: false "
                "explicitly if a float run is intended."
            ) from exc
        print(f"  Warning: QAT failed ({exc}). Training without quantization.")
        return model
