import tensorflow_model_optimization as tfmot
from src.models import create_nano_u
import tf_keras as keras

model = create_nano_u(input_shape=(60, 80, 3))
qat_model = tfmot.quantization.keras.quantize_model(model)

for layer in qat_model.layers:
    print(layer.name, layer.__class__.__name__)
