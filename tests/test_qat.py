import tensorflow as tf
from src.models import create_nano_u
from src.utils.qat import apply_qat_to_model

model = create_nano_u()
qat_model = apply_qat_to_model(model)

print("QAT Model layers:")
for layer in qat_model.layers:
    print(layer.__class__.__name__, layer.name)

