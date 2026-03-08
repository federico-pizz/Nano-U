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

    Uses the annotate-then-apply pattern so that layers TFMOT cannot quantize
    (UpSampling2D, MaxPooling2D, Concatenate, etc.) receive a NoOpQuantizeConfig
    instead of causing a hard failure.

    Args:
        model: A compiled or uncompiled Keras Functional model.

    Returns:
        A QAT-annotated model ready for training, or the original model if
        quantization fails for any reason (with a printed warning).
    """
    try:
        def _annotate_layer(layer):
            # 1. Identify if layer should be marked as "NoOp" (pass-through).
            # We check both the class name and the instance type for robustness.
            is_passthrough = (
                layer.__class__.__name__ in _QAT_PASSTHROUGH_NAMES or
                isinstance(layer, (keras.layers.Wrapper,)) # Don't re-annotate wrappers
            )

            try:
                if is_passthrough:
                    return tfmot.quantization.keras.quantize_annotate_layer(
                        layer, quantize_config=NoOpQuantizeConfig()
                    )
                # 2. Try default TFMOT annotation (works for Conv2D, Dense, etc.)
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            except Exception as e:
                # 3. If TFMOT rejects it (e.g. says "not a Layer instance" despite appearingly so),
                # just return the original layer. This is the ultimate fallback to avoid crashing.
                return layer

        annotated = keras.models.clone_model(
            model,
            clone_function=_annotate_layer,
        )

        with tfmot.quantization.keras.quantize_scope(
            {"NoOpQuantizeConfig": NoOpQuantizeConfig}
        ):
            qat_model = tfmot.quantization.keras.quantize_apply(annotated)

        print(f"  QAT applied: {qat_model.count_params():,} params "
              f"(was {model.count_params():,})")
        return qat_model

    except Exception as exc:
        print(f"  Warning: QAT failed ({exc}). Training without quantization.")
        return model
