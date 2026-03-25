import tensorflow as tf
from src.models import create_nano_u
import tensorflow_model_optimization as tfmot
wrapper = tfmot.quantization.keras.QuantizeWrapperV2

model = create_nano_u(input_shape=(60, 80, 3))

def _annotate(layer):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

annotated = tf.keras.models.clone_model(model, clone_function=_annotate)
qat_model = tfmot.quantization.keras.quantize_apply(annotated)

def check_layer(l):
    if isinstance(l, wrapper):
        return l.layer.__class__.__name__
    return l.__class__.__name__

classes = [check_layer(l) for l in qat_model.layers]
print(f"Number of BatchNormalization layers (clone_model API): {classes.count('BatchNormalization')}")

# Now check tfmot.quantization.keras.quantize_model
try:
    qat_model2 = tfmot.quantization.keras.quantize_model(model)
    classes2 = [check_layer(l) for l in qat_model2.layers]
    print(f"Number of BatchNormalization layers (quantize_model API): {classes2.count('BatchNormalization')}")
except Exception as e:
    print("Error doing quantize_model:", e)

