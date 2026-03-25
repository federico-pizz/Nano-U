import tensorflow as tf
from src.models import create_nano_u
import tensorflow_model_optimization as tfmot
from src.utils.qat import NoOpQuantizeConfig, _QAT_PASSTHROUGH_NAMES

model = create_nano_u(input_shape=(60, 80, 3))

def annotate(layer):
    if layer.__class__.__name__ in _QAT_PASSTHROUGH_NAMES:
        return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=NoOpQuantizeConfig())
    return layer

try:
    annotated = tf.keras.models.clone_model(model, clone_function=annotate)
    print("Clone model succeeded.")
    
    with tfmot.quantization.keras.quantize_scope({"NoOpQuantizeConfig": NoOpQuantizeConfig}):
        # In TFMOT, quantize_apply on a partially annotated model might quantize the rest? No.
        # But quantize_annotate_model can accept a partially annotated model and skip them? Let's see.
        final_annotated = tfmot.quantization.keras.quantize_annotate_model(annotated)
        qat_model = tfmot.quantization.keras.quantize_apply(final_annotated)
        
    wrapper = tfmot.quantization.keras.QuantizeWrapperV2
    def check_layer(l):
        if isinstance(l, wrapper):
            return l.layer.__class__.__name__
        return l.__class__.__name__

    classes = [check_layer(l) for l in qat_model.layers]
    print(f"Number of BatchNormalization layers: {classes.count('BatchNormalization')}")
except Exception as e:
    print(e)
