# API Reference

## Core Modules

### Training Pipeline

#### `src.train`

Main training interface for both teacher and student models.

```python
def train(model_name="nano_u", epochs=None, batch_size=None, lr=None,
          distill=False, teacher_weights=None, alpha=None, temperature=None,
          augment=True, config_path="config/config.yaml",
          enable_nas_monitoring=False, nas_log_dir=None, nas_csv_path=None,
          nas_log_freq='epoch', nas_monitor_batch_freq=10, nas_layer_selectors=None)
```

**Parameters:**
- `model_name` (str): "nano_u" or "bu_net"
- `epochs` (int): Training epochs (overrides config)
- `batch_size` (int): Batch size (overrides config)
- `lr` (float): Learning rate (overrides config)
- `distill` (bool): Enable knowledge distillation
- `teacher_weights` (str): Path to teacher model weights
- `alpha` (float): Student loss weight (0.0-1.0)
- `temperature` (float): Distillation temperature scaling
- `augment` (bool): Enable data augmentation
- `enable_nas_monitoring` (bool): Enable live NAS analysis
- `nas_layer_selectors` (list): Layer names to monitor

**Returns:**
- `model`: Trained model instance
- `history`: Training history object

**Example:**
```python
from src.train import train

# Basic student training with distillation
model, history = train(
    model_name="nano_u",
    epochs=100,
    distill=True,
    teacher_weights="models/bu_net.keras",
    enable_nas_monitoring=True,
    nas_layer_selectors=["encoder_conv_0", "bottleneck"]
)
```

#### `src.train.Distiller`

Knowledge distillation wrapper for teacher-student training.

```python
class Distiller(keras.Model):
    def __init__(self, student, teacher)
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, 
                alpha=0.5, temperature=2.0)
```

**Methods:**
- `train_step(data)`: Custom training step with distillation loss
- `test_step(data)`: Validation step
- `call(x, training=None)`: Forward pass through student model

### Model Architectures

#### `src.models.Nano_U.model_tf`

Ultra-lightweight student model for microcontroller deployment.

```python
class NanoU(Model):
    def __init__(self, n_channels=3, filters=None, bottleneck=64, 
                 decoder_filters=None, input_shape=(48, 64, 3))
```

**Architecture:**
- Encoder: [16, 32, 64] filters with depthwise separable convolutions
- Bottleneck: 64 filters
- Decoder: [32, 16] filters (no skip connections)
- Output: Single channel logits

**Methods:**
- `call(inputs, training=None)`: Forward pass
- `get_config()`: Model configuration for serialization

```python
def build_nano_u(input_shape=(48, 64, 3), n_channels=None, 
                 filters=None, bottleneck=64, decoder_filters=None)
```

**Example:**
```python
from src.models.Nano_U.model_tf import build_nano_u

model = build_nano_u(
    input_shape=(48, 64, 3),
    filters=[16, 32, 64],
    bottleneck=64,
    decoder_filters=[32, 16]
)
```

#### `src.models.BU_Net.model_tf`

Teacher model with U-Net architecture and skip connections.

```python
class BUNet(Model):
    def __init__(self, n_channels=3, name='BU_Net')
```

**Architecture:**
- Encoder: [64, 128, 256, 512, 1024, 2048] filters
- Bottleneck: 2048 filters
- Decoder: [1024, 512, 256, 128, 64] filters with skip connections
- Triple convolution blocks with depthwise separable convolutions

### Neural Architecture Search

#### `src.nas_covariance`

Real-time redundancy monitoring and NAS utilities.

```python
class NASMonitorCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, layer_selectors=None, log_dir="logs/nas",
                 csv_path="nas_metrics.csv", monitor_frequency="epoch", 
                 log_frequency=10)
```

**Parameters:**
- `validation_data`: Dataset for activation extraction
- `layer_selectors` (list): Layer names or regex patterns to monitor
- `log_dir` (str): TensorBoard logging directory
- `csv_path` (str): CSV output file path
- `monitor_frequency` (str): "epoch" or "batch"
- `log_frequency` (int): Batch frequency for monitoring

**Methods:**
- `on_epoch_end(epoch, logs)`: Compute and log NAS metrics
- `on_batch_end(batch, logs)`: Batch-level monitoring

**Example:**
```python
from src.nas_covariance import NASMonitorCallback

nas_callback = NASMonitorCallback(
    validation_data=val_ds,
    layer_selectors=["encoder_conv_0", "encoder_conv_1", "bottleneck"],
    log_dir="logs/nas",
    csv_path="logs/nas/nano_u_metrics.csv"
)
```

#### `src.nas_covariance.ActivationExtractor`

Cached extractor for intermediate layer activations.

```python
class ActivationExtractor:
    def __init__(self, model, layer_selectors)
    def __call__(self, inputs, training=False)
```

**Layer Selection:**
- Exact names: `["encoder_conv_0", "bottleneck"]`
- Regex patterns: `["/conv.*/", "/dense.*/"" ]`
- Layer instances: `[model.get_layer("conv1")]`

#### Redundancy Analysis Functions

```python
def covariance_redundancy(activations, data_format='channels_last')
```

**Parameters:**
- `activations` (Tensor): Layer activations [B, H, W, C] or [B, C]
- `data_format` (str): "channels_last" or "channels_first"

**Returns:**
- `redundancy_score` (float): 0.0-1.0 redundancy metric
- `condition_number` (float): Matrix conditioning indicator
- `mean_correlation` (float): Average feature correlation
- `trace` (float): Covariance matrix trace

**Interpretation:**
- `redundancy_score > 0.7`: High redundancy, reduce filters by 25-30%
- `redundancy_score 0.5-0.7`: Moderate-high, reduce by 15-20%
- `redundancy_score 0.3-0.5`: Moderate, optional 10-15% reduction
- `redundancy_score < 0.3`: Low redundancy, well-sized

### Data Pipeline

#### `src.utils.data`

Dataset loading and preprocessing utilities.

```python
def make_dataset(image_files, mask_files, batch_size=8, shuffle=True, 
                 augment=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 flip_prob=0.5, max_rotation_deg=15, brightness=0.2, 
                 contrast=0.2, saturation=0.2, hue=0.1)
```

**Parameters:**
- `image_files` (list): Paths to input images
- `mask_files` (list): Paths to segmentation masks
- `batch_size` (int): Batch size for training
- `shuffle` (bool): Shuffle dataset
- `augment` (bool): Enable data augmentation
- `mean`, `std` (list): Normalization parameters
- Augmentation parameters for rotation, color jitter

**Returns:**
- `tf.data.Dataset`: Preprocessed dataset ready for training

### Evaluation and Metrics

#### `src.utils.metrics`

Custom metrics for segmentation evaluation.

```python
class BinaryIoU(keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='binary_iou', **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None)
    def result(self)
```

**Parameters:**
- `threshold` (float): Probability threshold for binary prediction
- `name` (str): Metric name for logging

### Quantization and Deployment

#### `src.quantize`

Model quantization for edge deployment.

```python
def quantize_model(model_path, output_path, representative_dataset=None,
                   input_type='int8', output_type='int8')
```

**Parameters:**
- `model_path` (str): Path to trained Keras model
- `output_path` (str): Output path for quantized TFLite model
- `representative_dataset`: Representative data for calibration
- `input_type`, `output_type` (str): Quantization data types

**Example:**
```python
from src.quantize import quantize_model

quantize_model(
    model_path="models/nano_u.keras",
    output_path="models/nano_u_int8.tflite",
    representative_dataset=calibration_ds
)
```

#### `src.evaluate`

Model evaluation utilities.

```python
def evaluate_model(model_path, test_data_dir, batch_size=32)
```

**Returns:**
- Dictionary with metrics: IoU, accuracy, inference time

### Configuration Management

#### `src.utils.config`

Configuration loading and validation.

```python
def load_config(config_path="config/config.yaml")
```

**Returns:**
- Dictionary with model, training, and data configurations

**Configuration Structure:**
```yaml
data:
  input_shape: [48, 64, 3]
  normalization:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]

models:
  nano_u:
    filters: [16, 32, 64]
    bottleneck: 64
    decoder_filters: [32, 16]

training:
  nano_u:
    epochs: 100
    batch_size: 8
    learning_rate: 1e-4
    distillation:
      alpha: 0.3
      temperature: 4.0

nas:
  enabled: false
  layer_selectors: null
  log_freq: "epoch"
```

## Command Line Interface

### Training Scripts

```bash
# Basic training
python src/train.py --model nano_u --epochs 100

# Knowledge distillation
python src/train.py --model nano_u --distill --teacher-weights models/bu_net.keras

# With NAS monitoring
python src/train.py --model nano_u --enable-nas --nas-layers "encoder_conv_0,bottleneck"

# Custom hyperparameters
python src/train.py --model nano_u --lr 1e-4 --batch-size 8 --alpha 0.3 --temperature 4.0
```

### Experiment Runner

```bash
# Run hyperparameter sweep
python scripts/run_experiments.py --phase 4.1 --output results/phase_4_1/

# Resume from checkpoint
python scripts/run_experiments.py --phase 4.1 --resume --checkpoint results/checkpoint.json
```

### Model Operations

```bash
# Quantize model
python src/quantize.py --model-name nano_u --output models/nano_u_int8.tflite

# Evaluate model
python src/evaluate.py --model-name nano_u --out results/metrics.json

# Run inference demo
python scripts/infer_demo.py --model models/nano_u.keras --image test.png
```

## Error Handling

### Common Issues

**Model Instantiation Error:**
```python
TypeError: InternalError.__init__() missing 2 required positional arguments: 'op' and 'message'
```
- **Cause**: TensorFlow/Keras compatibility issue with custom model subclassing
- **Solution**: Use functional API or update TensorFlow version

**NAS Negative Condition Numbers:**
```
condition_number: -55814612.166666664
```
- **Cause**: Numerical instability in covariance computation
- **Solution**: Add matrix conditioning checks and regularization

**Layer Not Found:**
```python
ValueError: Layer 'encoder_conv_0' not found in model
```
- **Solution**: Use `model.summary()` to list available layer names

### Debugging Tools

```python
# List model layers
model = build_nano_u()
print([layer.name for layer in model.layers])

# Validate NAS selectors
from src.nas_covariance import ActivationExtractor
extractor = ActivationExtractor(model, ["encoder_conv_0"])
print(extractor.layer_names)

# Check data pipeline
ds = make_dataset(img_files, mask_files, batch_size=1)
for batch in ds.take(1):
    print(f"Input shape: {batch[0].shape}")
    print(f"Mask shape: {batch[1].shape}")
```

---

**Last Updated**: 2026-02-04  
**Version**: Research Preview