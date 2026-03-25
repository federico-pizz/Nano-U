"""Quantization-Aware Training (QAT) utilities for Keras models."""

import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot


# Layers that TFMOT cannot quantize: they pass data through without learnable
# weights and therefore need no fake-quantization nodes.
# Using string names for more robust matching across Keras namespace variants.
_QAT_PASSTHROUGH_NAMES = {
    "UpSampling2D",
    "MaxPooling2D",
    "AveragePooling2D",
    "GlobalAveragePooling2D",
    "Flatten",
    "Reshape",
    "Concatenate",
    "Add",
    "Dropout",
    "ZeroPadding2D",
    "Cropping2D",
    "InputLayer",
    "PadToMatch",  # Custom layer in builders.py
}


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Tells TFMOT to leave a layer completely unchanged during QAT.

    Used for layers that have no learnable weights (pooling, upsampling,
    concatenate, etc.) and therefore need no fake-quantization nodes.
    """

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def apply_qat_to_model(model: keras.Model) -> keras.Model:
    """Apply Quantization-Aware Training to a Keras Functional model.

    Uses `tfmot.quantization.keras.quantize_model` directly to ensure that
    layer patterns like Conv2D + BatchNormalization are correctly matched and
    folded during the fake-quantization process, which is critical for
    preserving performance.

    Args:
        model: A compiled or uncompiled Keras Functional model.

    Returns:
        A QAT-annotated model ready for training, or the original model if
        quantization fails for any reason (with a printed warning).
    """
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print(f"  QAT applied: {qat_model.count_params():,} params "
              f"(was {model.count_params():,})")
        return qat_model

    except Exception as exc:
        print(f"  Warning: QAT failed ({exc}). Training without quantization.")
        return model
