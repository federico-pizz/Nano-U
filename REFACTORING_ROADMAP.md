# Nano-U Refactoring Roadmap: From Complexity to Clarity

## 📋 Executive Summary

This document provides a comprehensive roadmap for refactoring the Nano-U codebase from its current complex state (~2000+ lines, 100% experiment failure rate) to a simplified, maintainable research framework suitable for microcontroller-based autonomous navigation research.

### Critical Issues Identified
- **100% experiment failure rate** due to Keras model serialization issues
- **NAS numerical instability** with negative condition numbers (-55M, -22M)
- **Excessive code complexity** with redundant training scripts and convoluted architecture
- **Model instantiation bugs** preventing any successful training runs

### Refactoring Goals
- **Reduce codebase from ~2000 to ~600 lines** (70% reduction)
- **Achieve functional model training** with proper error handling
- **Stabilize NAS computation** with numerical robustness
- **Simplify architecture** with clear separation of concerns
- **Enable rapid prototyping** for research iterations

---

## 🎯 Current State Analysis

### Code Complexity Breakdown
```
Current Structure (~2000+ lines):
├── src/train.py (411 lines) - Overly complex Distiller class
├── src/nas_covariance.py (991 lines) - Massive NAS implementation
├── src/models/Nano_U/model_tf.py (141 lines) - Serialization issues
├── scripts/run_experiments.py (409 lines) - Complex experiment runner
├── Multiple redundant training scripts
└── Scattered configuration management

Target Structure (~600 lines):
├── src/train.py (~200 lines) - Unified training pipeline
├── src/nas.py (~150 lines) - Simplified NAS computation
├── src/models.py (~150 lines) - Functional API models
├── src/experiment.py (~100 lines) - Streamlined experiments
└── Consolidated configuration system
```

### Critical Failures

#### 1. Model Instantiation Bug
```python
# Current Issue (from checkpoint.json):
"InternalError.__init__() missing 2 required positional arguments: 'message' and 'error_code'"

# Root Cause: Keras custom model subclassing conflicts
class NanoU(Model):  # Problematic inheritance
    def __init__(self, n_channels=3, filters=None, ...):
        super().__init__()  # Missing proper initialization
```

#### 2. NAS Numerical Instability
```python
# Current Issue (from nas_metrics.csv):
condition_number_conv2d_1: -55892632.0
condition_number_conv2d: -22336678.0

# Root Cause: Unstable covariance matrix computation
def covariance_redundancy(activations, eps=1e-8):
    cov_matrix = tf.linalg.experimental.movmean(...)  # Numerical instability
    condition_number = tf.linalg.det(cov_matrix)  # Can become negative
```

---

## 🏗️ Simplified Architecture Design

### Core Principles
1. **Functional over Object-Oriented**: Use Functional API instead of model subclassing
2. **Modular Components**: Clear separation between training, evaluation, and NAS
3. **Fail-Fast Design**: Early validation with comprehensive error handling
4. **Research-Oriented**: Optimized for experimentation rather than production

### New Project Structure
```
nano_u/
├── config/
│   └── experiments.yaml          # Single configuration file
├── src/
│   ├── models.py                 # All model definitions (Functional API)
│   ├── train.py                  # Unified training pipeline
│   ├── nas.py                    # Simplified NAS system
│   ├── data.py                   # Dataset preparation
│   ├── experiment.py             # Experiment runner
│   └── utils.py                  # Shared utilities
├── tests/
│   └── test_pipeline.py          # Comprehensive integration tests
├── scripts/
│   └── run_training.py           # Single training entry point
└── notebooks/
    └── research_analysis.ipynb   # Interactive analysis
```

---

## 🔧 Phase-by-Phase Implementation Plan

### Phase 1: Model Architecture Simplification (Days 1-3)

#### 1.1 Migrate to Functional API
**Current Problem**: Keras subclassing causes serialization issues
```python
# BEFORE (problematic):
@register_keras_serializable(package='Nano_U')
class NanoU(Model):
    def __init__(self, n_channels=3, filters=None, bottleneck=64, ...):
        super().__init__()  # Issues with custom serialization
        
# AFTER (simplified):
def create_nano_u(input_shape=(48, 64, 3), filters=None, bottleneck=64):
    """Build NanoU model using Functional API."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x1 = depthwise_sep_conv(inputs, filters[0])  # 16 filters
    x2 = depthwise_sep_conv(tf.nn.max_pool2d(x1), filters[1])  # 32 filters
    
    # Bottleneck
    bottleneck_out = depthwise_sep_conv(tf.nn.max_pool2d(x2), bottleneck)
    
    # Decoder
    up1 = tf.keras.layers.Conv2DTranspose(filters[1])(bottleneck_out)
    concat1 = tf.keras.layers.Concatenate()([up1, x2])
    
    up2 = tf.keras.layers.Conv2DTranspose(filters[0])(concat1)
    concat2 = tf.keras.layers.Concatenate()([up2, x1])
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(concat2)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='nano_u')
```

#### 1.2 Consolidate Layer Definitions
**Target**: Merge [`src/models/Nano_U/layers_tf.py`](src/models/Nano_U/layers_tf.py) and [`src/models/Nano_U/model_tf.py`](src/models/Nano_U/model_tf.py) into single [`src/models.py`](src/models.py)

```python
# src/models.py (new consolidated file)
def depthwise_sep_conv(x, filters, stride=1, name=None):
    """Depthwise separable convolution block."""
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', strides=stride)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def create_nano_u(input_shape=(48, 64, 3), filters=[16, 32], bottleneck=64):
    """Ultra-lightweight U-Net for microcontrollers."""
    # Implementation here...

def create_bu_net(input_shape=(48, 64, 3), filters=[32, 64, 128], bottleneck=256):
    """Teacher model for knowledge distillation."""
    # Implementation here...
```

### Phase 2: Training Pipeline Unification (Days 4-6)

#### 2.1 Simplify Distillation Training
**Current Problem**: [`src/train.py`](src/train.py:49) has overly complex [`Distiller`](src/train.py:49) class (132 lines)
```python
# BEFORE (complex):
class Distiller(keras.Model):
    def train_step(self, data):
        # 38 lines of complex logic with nested try-catch
        
# AFTER (simplified):
def train_step(student, teacher, x, y, optimizer, alpha=0.3, temperature=4.0):
    """Single training step with distillation."""
    with tf.GradientTape() as tape:
        # Forward pass
        teacher_pred = teacher(x, training=False)
        student_pred = student(x, training=True)
        
        # Compute losses
        student_loss = tf.keras.losses.binary_crossentropy(y, student_pred)
        distill_loss = tf.keras.losses.kl_divergence(
            tf.nn.softmax(teacher_pred / temperature),
            tf.nn.softmax(student_pred / temperature)
        ) * (temperature ** 2)
        
        total_loss = alpha * student_loss + (1 - alpha) * distill_loss
    
    # Apply gradients
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    
    return {
        'loss': total_loss,
        'student_loss': student_loss,
        'distillation_loss': distill_loss
    }
```

#### 2.2 Unified Training Function
```python
# src/train.py (simplified)
def train_model(config_path="config/experiments.yaml", experiment_name="default"):
    """Main training function with automatic teacher/student handling."""
    config = yaml.safe_load(open(config_path))[experiment_name]
    
    # Build models
    if config.get('use_distillation', False):
        teacher = create_bu_net(input_shape=config['input_shape'])
        teacher.load_weights(config['teacher_weights'])
        student = create_nano_u(input_shape=config['input_shape'])
        return train_with_distillation(student, teacher, config)
    else:
        model = create_nano_u(input_shape=config['input_shape'])
        return train_single_model(model, config)
```

### Phase 3: NAS System Redesign (Days 7-9)

#### 3.1 Numerical Stability Fix
**Current Problem**: [`src/nas_covariance.py`](src/nas_covariance.py:229) produces negative condition numbers
```python
# BEFORE (unstable):
def covariance_redundancy(activations, eps=1e-8):
    cov_matrix = tf.linalg.experimental.movmean(...)  # Numerical issues
    condition_number = tf.linalg.det(cov_matrix)      # Can be negative!

# AFTER (stable):
def compute_layer_redundancy(activations, eps=1e-6):
    """Stable redundancy computation using SVD decomposition."""
    # Reshape to (samples, features)
    reshaped = tf.reshape(activations, (-1, tf.shape(activations)[-1]))
    
    # Centered covariance
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
        'redundancy_score': redundancy,
        'condition_number': condition_number,
        'rank': tf.reduce_sum(tf.cast(singular_values > eps, tf.float32))
    }
```

#### 3.2 Simplified NAS Callback
**Target**: Reduce [`src/nas_covariance.py`](src/nas_covariance.py:587) from 990 lines to ~150 lines
```python
# src/nas.py (new simplified file)
class NASCallback(tf.keras.callbacks.Callback):
    """Lightweight NAS monitoring with stable computation."""
    
    def __init__(self, layers_to_monitor=['conv2d', 'conv2d_1'], 
                 log_frequency=10):
        super().__init__()
        self.layers_to_monitor = layers_to_monitor
        self.log_frequency = log_frequency
        self.metrics = defaultdict(list)
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency != 0:
            return
            
        # Extract activations
        for layer_name in self.layers_to_monitor:
            layer = self.model.get_layer(layer_name)
            # Simple activation extraction without complex wrapping
            activations = layer.output
            redundancy = compute_layer_redundancy(activations)
            self.metrics[f'{layer_name}_redundancy'].append(redundancy['redundancy_score'])
    
    def save_metrics(self, filepath):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
```

### Phase 4: Configuration and Experiment Management (Days 10-11)

#### 4.1 Unified Configuration System
**Current Problem**: Configuration scattered across multiple files
```yaml
# config/experiments.yaml (single source of truth)
default:
  model_name: "nano_u"
  input_shape: [48, 64, 3]
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  
  # Architecture
  filters: [16, 32]
  bottleneck: 64
  
  # Training
  use_distillation: false
  teacher_weights: null
  
  # Data
  dataset_path: "data/segmentation"
  validation_split: 0.2

distillation_experiment:
  <<: *default
  use_distillation: true
  teacher_weights: "models/bu_net_weights.h5"
  alpha: 0.3
  temperature: 4.0
  
  # Distillation-specific training
  epochs: 100
  learning_rate: 0.0005

nas_experiment:
  <<: *default
  use_nas: true
  nas_frequency: 10
  layers_to_monitor: ['conv2d', 'conv2d_1', 'conv2d_2']
```

#### 4.2 Streamlined Experiment Runner
**Target**: Replace [`scripts/run_experiments.py`](scripts/run_experiments.py:28) (409 lines) with ~100 lines
```python
# src/experiment.py (simplified)
def run_experiment(config_name, output_dir="results"):
    """Run single experiment with comprehensive logging."""
    try:
        # Load configuration
        config = load_config(config_name)
        
        # Setup logging
        experiment_dir = Path(output_dir) / f"{config_name}_{datetime.now():%Y%m%d_%H%M%S}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and training
        model = create_model_from_config(config)
        history = train_model_with_config(model, config)
        
        # Save results
        model.save(experiment_dir / "model.h5")
        save_training_history(history, experiment_dir / "history.json")
        
        return {
            'status': 'success',
            'final_metrics': history.history,
            'model_path': str(experiment_dir / "model.h5")
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def run_experiment_sweep(experiment_configs):
    """Run multiple experiments with parallel processing."""
    results = []
    for config_name in experiment_configs:
        result = run_experiment(config_name)
        results.append(result)
        
        # Early stopping on repeated failures
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        if failed_count > len(results) * 0.5:  # Stop if >50% fail
            print(f"Stopping sweep due to high failure rate: {failed_count}/{len(results)}")
            break
    
    return results
```

### Phase 5: Testing and Validation (Days 12-13)

#### 5.1 Integration Test Suite
```python
# tests/test_pipeline.py (comprehensive)
def test_model_instantiation():
    """Test that models can be created and compiled without errors."""
    model = create_nano_u()
    assert model is not None
    assert len(model.layers) > 0
    
    # Test compilation
    model.compile(optimizer='adam', loss='binary_crossentropy')
    assert model.compiled_loss is not None

def test_training_pipeline():
    """Test full training pipeline with synthetic data."""
    # Generate synthetic data
    x = np.random.random((10, 48, 64, 3))
    y = np.random.random((10, 48, 64, 1))
    
    # Test training
    model = create_nano_u()
    history = train_single_model(model, {
        'epochs': 2,
        'batch_size': 4,
        'learning_rate': 0.001
    }, (x, y), (x, y))
    
    assert 'loss' in history.history
    assert len(history.history['loss']) == 2

def test_nas_computation():
    """Test NAS computation stability."""
    # Create dummy activations
    activations = tf.random.normal((32, 24, 32, 64))  # (batch, h, w, channels)
    
    # Compute redundancy
    redundancy = compute_layer_redundancy(activations)
    
    # Verify outputs are reasonable
    assert 0.0 <= redundancy['redundancy_score'] <= 1.0
    assert redundancy['condition_number'] > 0
    assert redundancy['rank'] > 0
```

#### 5.2 Performance Benchmarks
```python
def test_model_size_constraints():
    """Verify model meets microcontroller constraints."""
    model = create_nano_u()
    param_count = model.count_params()
    
    # Verify parameter count constraint
    assert param_count < 50000, f"Model too large: {param_count} parameters"
    
    # Test quantization compatibility
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Verify quantized model size
    model_size = len(tflite_model)
    assert model_size < 200000, f"Quantized model too large: {model_size} bytes"
```

---

## 📈 Migration Strategy

### Week 1: Foundation Refactoring
1. **Day 1-2**: Model architecture migration to Functional API
2. **Day 3-4**: Training pipeline simplification
3. **Day 5-6**: NAS system redesign and numerical stability fixes
4. **Day 7**: Integration testing and validation

### Week 2: System Integration
1. **Day 8-9**: Configuration system unification
2. **Day 10-11**: Experiment runner streamlining
3. **Day 12**: Comprehensive testing and benchmarking
4. **Day 13-14**: Documentation update and validation

### Migration Steps
1. **Backup Current Code**: Create branch `backup/complex-version`
2. **Parallel Development**: Develop simplified version in `feature/simplified-architecture`
3. **Progressive Testing**: Test each component independently
4. **Integration Validation**: Ensure all core functionalities work
5. **Performance Verification**: Confirm no regression in model quality
6. **Documentation Update**: Update all documentation to reflect changes

---

## 🎯 Success Metrics

### Technical Metrics
- **Code Reduction**: From ~2000 to ~600 lines (70% reduction achieved)
- **Experiment Success Rate**: From 0% to >90% success rate
- **NAS Stability**: Positive condition numbers, stable redundancy scores
- **Model Quality**: Maintain IoU performance within 5% of current baseline

### Research Enablement Metrics
- **Training Time**: <10 minutes for basic experiments
- **Configuration Changes**: Single YAML edit for new experiments
- **Error Recovery**: Clear error messages with actionable solutions
- **ESP32 Compatibility**: Successful quantization and deployment

### Development Metrics
- **Test Coverage**: >90% line coverage
- **CI/CD Pipeline**: All tests pass automatically
- **Documentation**: Complete API reference and usage examples
- **Developer Onboarding**: New contributor can run experiments in <30 minutes

---

## ⚠️ Risk Mitigation

### High-Risk Areas
1. **Model Performance Regression**: Functional API might affect training dynamics
   - **Mitigation**: Extensive validation with existing baselines
   - **Rollback Plan**: Keep original models for comparison

2. **NAS Algorithm Changes**: Mathematical modifications could affect research results
   - **Mitigation**: A/B test new NAS against reference implementation
   - **Documentation**: Clear mathematical derivations for all changes

3. **Configuration Breaking Changes**: YAML format changes might break existing experiments
   - **Mitigation**: Provide migration script for old configurations
   - **Backward Compatibility**: Support old format during transition period

### Quality Assurance
- **Automated Testing**: Run full test suite on every commit
- **Performance Monitoring**: Track training time and memory usage
- **Model Validation**: Compare outputs between old and new implementations
- **Documentation Reviews**: Ensure all changes are properly documented

---

## 📚 Implementation Resources

### Key Files to Create
```
REFACTORING_ROADMAP.md         # This document
src/models.py                  # Consolidated model definitions
src/train.py                   # Simplified training pipeline
src/nas.py                     # Redesigned NAS system
src/experiment.py              # Streamlined experiment runner
config/experiments.yaml        # Unified configuration
tests/test_pipeline.py         # Comprehensive test suite
scripts/migrate_config.py      # Configuration migration tool
```

### Development Tools
- **Code Analysis**: Use `pylint` and `black` for consistent formatting
- **Performance Profiling**: `cProfile` for identifying bottlenecks
- **Memory Monitoring**: `memory_profiler` for memory usage tracking
- **Model Visualization**: `tensorflow.keras.utils.plot_model` for architecture diagrams

### Reference Implementations
- **Functional API Examples**: TensorFlow official documentation
- **Knowledge Distillation**: Hinton et al. 2015 reference implementation
- **NAS Stability**: Neural Architecture Search without Training papers
- **ESP32 Deployment**: TensorFlow Lite Micro examples

---

## 🚀 Expected Outcomes

### Immediate Benefits (Week 1)
- **Functional Training**: Experiments actually complete successfully
- **Stable NAS**: Consistent, meaningful redundancy metrics
- **Clear Architecture**: Simplified, understandable codebase

### Research Acceleration (Week 2)
- **Rapid Prototyping**: Quick iteration on model architectures
- **Reliable Experiments**: Consistent, reproducible results
- **Easy Configuration**: Simple parameter sweeps and ablation studies

### Long-term Impact (Month 1+)
- **Maintainable Research**: Easy to extend and modify
- **Collaborative Development**: Clear structure for multiple contributors
- **Publication Ready**: Clean, documented implementation for papers

### Success Indicators
✅ **All experiments complete without errors**  
✅ **NAS metrics are numerically stable and meaningful**  
✅ **Model training time reduced by >50%**  
✅ **Codebase is understandable by new contributors**  
✅ **ESP32 deployment pipeline works end-to-end**  

This refactoring roadmap provides a clear path from the current complex, non-functional codebase to a simplified, research-oriented framework that enables rapid experimentation with ultra-low-power CNNs for microcontroller-based autonomous navigation.