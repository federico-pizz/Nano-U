# Nano-U Refactoring Documentation

## Overview

This document provides comprehensive documentation for the refactored Nano-U codebase,
including migration guides, API reference, and usage examples.

## Migration Guide

### From Old to New Architecture

#### 1. Model Definitions

**Old (problematic):**
```python
@register_keras_serializable(package='Nano_U')
class NanoU(Model):
    def __init__(self, n_channels=3, filters=None, bottleneck=64, ...):
        super().__init__()  # Issues with custom serialization
        # Complex initialization logic
```

**New (simplified):**
```python
def create_nano_u(
    input_shape: Tuple[int, int, int] = (48, 64, 3),
    filters: List[int] = [16, 32],
    bottleneck: int = 64,
    name: str = 'nano_u'
) -> Model:
    """Create ultra-lightweight U-Net for microcontroller deployment."""
    inputs = layers.Input(shape=input_shape, name='input_image')
    # Functional API implementation
    return Model(inputs=inputs, outputs=outputs, name=name)
```

#### 2. Training Pipeline

**Old (complex):**
```python
class Distiller(keras.Model):
    def train_step(self, data):
        # 38 lines of complex logic with nested try-catch
        # Manual gradient computation
        # Complex loss calculation
```

**New (simplified):**
```python
def train_step(student: keras.Model, teacher: Optional[keras.Model], 
               x: tf.Tensor, y: tf.Tensor, optimizer: keras.optimizers.Optimizer,
               alpha: float = 0.3, temperature: float = 4.0) -> Dict[str, tf.Tensor]:
    """Single training step with optional knowledge distillation."""
    with tf.GradientTape() as tape:
        # Forward pass
        if teacher is not None:
            teacher_pred = teacher(x, training=False)
        student_pred = student(x, training=True)
        
        # Compute losses
        student_loss = tf.keras.losses.binary_crossentropy(y, student_pred)
        
        if teacher is not None:
            distill_loss = tf.keras.losses.kl_divergence(
                tf.nn.softmax(teacher_pred / temperature),
                tf.nn.softmax(student_pred / temperature)
            ) * (temperature ** 2)
            total_loss = alpha * student_loss + (1 - alpha) * distill_loss
        else:
            total_loss = student_loss
    
    # Apply gradients
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    
    return {
        'loss': total_loss,
        'student_loss': student_loss,
        'distillation_loss': distill_loss
    }
```

#### 3. NAS System

**Old (unstable):**
```python
def covariance_redundancy(activations, eps=1e-8):
    cov_matrix = tf.linalg.experimental.movmean(...)  # Numerical issues
    condition_number = tf.linalg.det(cov_matrix)      # Can be negative!
```

**New (stable):**
```python
def compute_layer_redundancy(activations: tf.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Compute stable redundancy score using SVD decomposition."""
    # Reshape to (samples, features) for SVD
    reshaped = tf.reshape(activations, (-1, tf.shape(activations)[-1]))
    
    # Center activations
    mean_act = tf.reduce_mean(reshaped, axis=0)
    centered = reshaped - mean_act
    
    # SVD for numerical stability
    s, u, v = tf.linalg.svd(centered, full_matrices=False)
    singular_values = tf.maximum(s, eps)  # Clamp to avoid zeros
    
    # Condition number using SVD
    condition_number = tf.reduce_max(singular_values) / tf.reduce_min(singular_values)
    
    # Redundancy score (normalized)
    redundancy = 1.0 / (1.0 + tf.math.log(condition_number + 1.0))
    
    return {
        'redundancy_score': float(redundancy.numpy()),
        'condition_number': float(condition_number.numpy()),
        'rank': int(tf.reduce_sum(tf.cast(singular_values > eps, tf.int32)).numpy()),
        'num_channels': int(channels.numpy())
    }
```

### Configuration Changes

#### New Configuration Structure

```yaml
# config/experiments.yaml
experiments:
  # Standard training
  standard:
    <<: *training.standard
    model_name: "nano_u"
    use_distillation: false
    use_nas: false
    layers_to_monitor: ["conv2d", "conv2d_1"]
    nas_frequency: 10
    
  # Knowledge distillation training
  distillation:
    <<: *training.distillation
    model_name: "nano_u"
    use_distillation: true
    teacher_weights: "models/bu_net_weights.h5"
    alpha: 0.3
    temperature: 4.0
    epochs: 100
    learning_rate: 0.0005
```

#### Migration Script

```python
# scripts/migrate_config.py
import yaml
import shutil
from pathlib import Path

def migrate_old_config(old_path: str, new_path: str):
    """Migrate old configuration to new format."""
    with open(old_path) as f:
        old_config = yaml.safe_load(f)
    
    new_config = {
        "experiments": {
            "migrated": {
                "model_name": old_config.get("model", "nano_u"),
                "epochs": old_config.get("epochs", 50),
                "batch_size": old_config.get("batch_size", 16),
                "learning_rate": old_config.get("learning_rate", 0.001),
                "use_distillation": old_config.get("distillation", False),
                "teacher_weights": old_config.get("teacher_weights"),
                "alpha": old_config.get("alpha", 0.3),
                "temperature": old_config.get("temperature", 4.0),
                "use_nas": old_config.get("nas", False),
                "layers_to_monitor": old_config.get("layers_to_monitor", ["conv2d", "conv2d_1"]),
                "nas_frequency": old_config.get("nas_frequency", 10)
            }
        }
    }
    
    with open(new_path, 'w') as f:
        yaml.dump(new_config, f)
    
    print(f"Configuration migrated to: {new_path}")
```

## API Reference

### Models Module (`src/models.py`)

#### `create_nano_u()`
```python
def create_nano_u(
    input_shape: Tuple[int, int, int] = (48, 64, 3),
    filters: List[int] = [16, 32],
    bottleneck: int = 64,
    name: str = 'nano_u'
) -> Model:
    """Create ultra-lightweight U-Net for microcontroller deployment.
    
    Args:
        input_shape: Input shape (height, width, channels)
        filters: List of encoder/decoder filter sizes
        bottleneck: Number of filters in bottleneck layer
        name: Model name
    
    Returns:
        Compiled Keras Functional API model
    """
```

#### `create_bu_net()`
```python
def create_bu_net(
    input_shape: Tuple[int, int, int] = (48, 64, 3),
    filters: List[int] = [32, 64, 128],
    bottleneck: int = 256,
    name: str = 'bu_net'
) -> Model:
    """Create teacher model (BU_Net) for knowledge distillation.
    
    Args:
        input_shape: Input shape (height, width, channels)
        filters: List of encoder/decoder filter sizes
        bottleneck: Number of filters in bottleneck layer
        name: Model name
    
    Returns:
        Compiled Keras Functional API model
    """
```

#### `create_model_from_config()`
```python
def create_model_from_config(config: dict) -> Model:
    """Create model based on configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Compiled Keras model
    """
```

### Training Module (`src/train.py`)

#### `train_step()`
```python
def train_step(student: keras.Model, teacher: Optional[keras.Model], 
               x: tf.Tensor, y: tf.Tensor, optimizer: keras.optimizers.Optimizer,
               alpha: float = 0.3, temperature: float = 4.0) -> Dict[str, tf.Tensor]:
    """Single training step with optional knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Optional teacher model for distillation
        x: Input batch
        y: Ground truth labels
        optimizer: Optimizer to use
        alpha: Distillation weight (0 = no distillation, 1 = pure distillation)
        temperature: Temperature for distillation
    
    Returns:
        Dictionary of loss components
    """
```

#### `train_single_model()`
```python
def train_single_model(model: keras.Model, config: Dict, 
                      train_data: Tuple[tf.Tensor, tf.Tensor],
                      val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> keras.callbacks.History:
    """Train a single model without distillation.
    
    Args:
        model: Model to train
        config: Training configuration
        train_data: Training data (x, y)
        val_data: Optional validation data (x, y)
    
    Returns:
        Training history
    """
```

#### `train_with_distillation()`
```python
def train_with_distillation(student: keras.Model, teacher: keras.Model, config: Dict,
                            train_data: Tuple[tf.Tensor, tf.Tensor],
                            val_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> keras.callbacks.History:
    """Train student model with knowledge distillation from teacher.
    
    Args:
        student: Student model to train
        teacher: Teacher model for distillation
        config: Training configuration
        train_data: Training data (x, y)
        val_data: Optional validation data (x, y)
    
    Returns:
        Training history
    """
```

#### `train_model()`
```python
def train_model(config_path: str = "config/experiments.yaml", 
                experiment_name: str = "default") -> Dict[str, any]:
    """Main training function with automatic teacher/student handling.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name of experiment to run
    
    Returns:
        Dictionary with training results and status
    """
```

### NAS Module (`src/nas.py`)

#### `compute_layer_redundancy()`
```python
def compute_layer_redundancy(activations: tf.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Compute stable redundancy score using SVD decomposition.
    
    Args:
        activations: Layer activations tensor (batch, height, width, channels)
        eps: Small value to prevent division by zero
    
    Returns:
        Dictionary with redundancy metrics
    """
```

#### `NASCallback()`
```python
class NASCallback(tf.keras.callbacks.Callback):
    """Lightweight NAS monitoring callback with stable computation.
    
    Monitors specified layers during training and logs redundancy metrics.
    """
    
    def __init__(self, layers_to_monitor: List[str] = None, log_frequency: int = 10,
                 output_dir: str = 'nas_logs/'):
        """Initialize NAS callback.
        
        Args:
            layers_to_monitor: List of layer names to monitor
            log_frequency: Log every N batches
            output_dir: Directory to save NAS logs
        """
```

## Usage Examples

### Basic Training

```python
from src.train import train_model

# Run standard training
result = train_model(
    config_path="config/experiments.yaml",
    experiment_name="standard"
)

if result["status"] == "success":
    print(f"Model saved to: {result["model_path"]}")
    print(f"Final metrics: {result["final_metrics"]}")
```

### Knowledge Distillation

```python
from src.train import train_model

# Run distillation training
result = train_model(
    config_path="config/experiments.yaml",
    experiment_name="distillation"
)

if result["status"] == "success":
    print(f"Student model saved to: {result["model_path"]}")
    print(f"Distillation metrics: {result["final_metrics"]}")
```

### NAS Monitoring

```python
from src.train import train_model
from src.nas import analyze_model_redundancy

# Run training with NAS monitoring
result = train_model(
    config_path="config/experiments.yaml",
    experiment_name="nas"
)

if result["status"] == "success":
    # Analyze redundancy after training
    model = keras.models.load_model(result["model_path"])
    x_sample = np.random.random((1, 48, 64, 3)).astype(np.float32)
    
    analysis = analyze_model_redundancy(model, x_sample)
    print("NAS Analysis:")
    print(analysis["aggregate"])
```

### Custom Training Loop

```python
from src.models import create_nano_u
from src.train import train_step
from src.nas import NASCallback
import tensorflow as tf

# Create model and optimizer
model = create_nano_u()
optimizer = tf.keras.optimizers.Adam(0.001)

# Create callback
callback = NASCallback(layers_to_monitor=["conv2d", "conv2d_1"], log_frequency=10)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        # Training step
        losses = train_step(model, None, x_batch, y_batch, optimizer)
        
        # Callback
        callback.on_train_batch_end(batch)
    
    print(f"Epoch {epoch+1}: Loss = {losses[\"loss\"].numpy():.4f}")

# Save model
model.save("custom_trained_model.h5")
```

## Performance Benchmarks

### Training Speed

```python
import time
from src.models import create_nano_u
from src.train import train_single_model

# Test training speed
model = create_nano_u()

# Create synthetic data
x_train = np.random.random((100, 48, 64, 3)).astype(np.float32)
 y_train = np.random.randint(0, 2, (100, 48, 64, 1)).astype(np.float32)

# Time training
start_time = time.time()
history = train_single_model(
    model, 
    {"epochs": 5, "batch_size": 16, "learning_rate": 0.001},
    (x_train, y_train)
)
end_time = time.time()

training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")
print(f"Speed: {len(x_train) * 5 / training_time:.2f} samples/second")
```

### Model Size Analysis

```python
from src.models import create_nano_u, count_parameters
from src.train import train_single_model

# Create and analyze model
model = create_nano_u()
param_count = count_parameters(model)

print(f"Model parameters: {param_count:,}")
print(f"Model size estimate: {param_count * 4 / 1024 / 1024:.2f} MB")

# Test quantization compatibility
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    quantized_size = len(tflite_model) / 1024
    print(f"Quantized model size: {quantized_size:.2f} KB")
except Exception as e:
    print(f"Quantization failed: {e}")
```

## Troubleshooting

### Common Issues

#### 1. GPU Memory Issues

```python
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth failed: {e}")
```

#### 2. Model Serialization Issues

```python
# Ensure proper model saving
model = create_nano_u()
model.save("model.h5", save_format='h5')

# For TensorFlow SavedModel format
try:
    model.save("model_saved_model", save_format='tf')
except Exception as e:
    print(f"SavedModel format failed: {e}")
    # Fall back to HDF5
    model.save("model.h5", save_format='h5')
```

#### 3. NAS Computation Issues

```python
# Validate NAS computation
from src.nas import validate_nas_computation

if validate_nas_computation():
    print("NAS computation is working correctly")
else:
    print("NAS computation has issues - check input data and layer names")
```

### Debug Mode

```python
# Enable debug mode for detailed logging
config = {
    "model_name": "nano_u",
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 0.001,
    "debug_mode": True,  # Enable debug mode
    "verbose_logging": True
}

result = train_model(config_path="config/experiments.yaml", experiment_name="debug_test")
```

## Best Practices

1. **Use Functional API**: Always use Functional API for model definitions to ensure proper serialization
2. **Monitor NAS**: Enable NAS monitoring during training to identify redundant layers
3. **Validate Data**: Always validate input data shapes and normalization
4. **Test Incrementally**: Test each component (models, training, NAS) separately before integration
5. **Use Checkpoints**: Enable model checkpoints to prevent training loss
6. **Profile Performance**: Monitor training time and memory usage for optimization
7. **Document Experiments**: Keep detailed records of experiment configurations and results

## Version History

### Version 1.0.0 (Current)
- Complete refactoring to Functional API
- Simplified training pipeline
- Stable NAS system with SVD decomposition
- Unified configuration system
- Comprehensive test suite
- Streamlined experiment runner

### Migration Notes
- All model definitions moved to `src/models.py`
- Training pipeline simplified in `src/train.py`
- NAS system redesigned in `src/nas.py`
- Configuration unified in `config/experiments.yaml`
- Experiment runner streamlined in `scripts/run_experiments.py`
- Comprehensive tests in `tests/test_pipeline.py`

## Support

For issues and questions:
1. Check troubleshooting section
2. Review test cases for examples
3. Validate configuration files
4. Ensure TensorFlow version compatibility
5. Check GPU/CPU compatibility

---

*This documentation is automatically generated and maintained as part of the refactoring process.*