# Nano-U Usage Examples and Best Practices

This document provides practical examples for using the Nano-U project, from basic training to advanced NAS monitoring and architecture optimization.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [NAS Monitoring](#nas-monitoring)
4. [Visualization](#visualization)
5. [Pipeline Orchestration](#pipeline-orchestration)
6. [Metric Interpretation](#metric-interpretation)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Training (No NAS)
```bash
# Train Nano_U with default config
python src/train.py --model nano_u --epochs 10

# Train BU_Net (teacher model)
python src/train.py --model bu_net --epochs 10
```

### Training with Knowledge Distillation
```bash
# Step 1: Train teacher (bu_net)
python src/train.py --model bu_net --epochs 50

# Step 2: Distill student (nano_u) from teacher
python src/train.py --model nano_u --epochs 50 --distill \
    --teacher-weights models/bu_net.keras
```

---

## Training Models

### Basic Training Options

#### Configure Hyperparameters via CLI
```bash
python src/train.py \
    --model nano_u \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001
```

#### Use Custom Config File
```bash
python src/train.py \
    --model nano_u \
    --config config/custom_config.yaml
```

#### Disable Data Augmentation
```bash
python src/train.py \
    --model nano_u \
    --no-augment
```

### Knowledge Distillation Options

#### Standard Distillation
```bash
python src/train.py \
    --model nano_u \
    --distill \
    --teacher-weights models/bu_net.keras \
    --alpha 0.3 \
    --temperature 4.0
```

**Parameters**:
- `--alpha`: Weight for student loss (lower = more teacher influence). Range: [0, 1]
- `--temperature`: Softening parameter for teacher outputs (higher = softer). Typical: 2-5

---

## NAS Monitoring

### Overview

NAS (Neural Architecture Search) monitoring analyzes feature redundancy during training to identify overparameterization.

### Basic NAS Monitoring (Output Only)

```bash
# Monitor model output (may return zeros for single-channel output)
python src/train.py \
    --model nano_u \
    --epochs 10 \
    --enable-nas
```

**Warning**: For segmentation models with 1-channel output, this returns zero metrics. Use internal layer monitoring instead.

### Internal Layer Monitoring (Recommended)

#### Method 1: CLI Arguments
```bash
# Monitor specific layers
python src/train.py \
    --model nano_u \
    --epochs 10 \
    --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,encoder_conv_2,bottleneck"
```

#### Method 2: Config File
Edit `config/config.yaml`:
```yaml
nas:
  enabled: true
  layer_selectors: ['encoder_conv_0', 'encoder_conv_1', 'encoder_conv_2', 'bottleneck']
```

Then run:
```bash
python src/train.py --model nano_u --epochs 10 --enable-nas
```

#### Method 3: Regex Pattern
```bash
# Monitor all layers with 'conv' in the name
python src/train.py \
    --model nano_u \
    --epochs 10 \
    --enable-nas \
    --nas-layers "/conv.*/"
```

### Custom NAS Configuration

```bash
python src/train.py \
    --model nano_u \
    --epochs 10 \
    --enable-nas \
    --nas-layers "encoder_conv_0,bottleneck" \
    --nas-log-dir logs/nas_experiment_1 \
    --nas-csv-path results/nas_metrics_nano_u.csv \
    --nas-log-freq epoch \
    --nas-batch-freq 1
```

**Parameters**:
- `--nas-layers`: Comma-separated layer names or regex patterns
- `--nas-log-dir`: Directory for TensorBoard logs
- `--nas-csv-path`: Path for CSV output
- `--nas-log-freq`: `epoch` or `batch`
- `--nas-batch-freq`: Frequency for batch-level monitoring

---

## Visualization

### Generate NAS Plots from Training

```bash
# After training with NAS
python src/plot_nas_metrics.py --csv logs/nas/nano_u_nas_metrics.csv
```

**Outputs** (saved to `results/nas_plots/`):
1. `redundancy_over_time.png` - Redundancy trend
2. `correlation_analysis.png` - Feature correlation with interpretation zones
3. `nas_dashboard.png` - 4-panel dashboard with summary stats
4. `model_comparison.png` - (when comparing multiple models)

### Compare Multiple Models

```bash
# Train multiple models with NAS
python src/train.py --model bu_net --epochs 10 --enable-nas
mv logs/nas/bu_net_nas_metrics.csv results/bu_net_nas.csv

python src/train.py --model nano_u --epochs 10 --enable-nas
mv logs/nas/nano_u_nas_metrics.csv results/nano_u_nas.csv

# Generate comparison plot
python src/plot_nas_metrics.py \
    --compare results/bu_net_nas.csv results/nano_u_nas.csv \
    --model-names "BU_Net (Teacher)" "Nano_U (Student)" \
    --output-dir results/comparison
```

### Custom Output Directory

```bash
python src/plot_nas_metrics.py \
    --csv metrics.csv \
    --output-dir experiments/run_042/plots
```

---

## Pipeline Orchestration

### Full Training Pipeline

```bash
# Run complete pipeline: train teacher, distill student, quantize, evaluate
python scripts/tf_pipeline.py pipeline
```

### Pipeline with NAS Monitoring

```bash
# Enable NAS for teacher training
python scripts/tf_pipeline.py pipeline --enable-nas-for-teacher

# Enable NAS for both teacher and student
python scripts/tf_pipeline.py pipeline \
    --enable-nas-for-teacher \
    --enable-nas-for-student
```

### Individual Pipeline Commands

```bash
# Train teacher with NAS
python scripts/tf_pipeline.py train --model bu_net --enable-nas

# Distill student with NAS and layer monitoring
python scripts/tf_pipeline.py train \
    --model nano_u \
    --distill \
    --teacher-weights models/bu_net.keras \
    --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,bottleneck"

# Quantize model
python scripts/tf_pipeline.py quantize \
    --model-name nano_u \
    --output models/nano_u_int8.tflite

# Evaluate model
python scripts/tf_pipeline.py eval \
    --model-name nano_u \
    --out results/nano_u_metrics.json
```

---

## Metric Interpretation

### Redundancy Score

**What it measures**: Off-diagonal covariance (feature correlation)

| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| > 0.7 | **High redundancy** | Reduce filters by 25-30% |
| 0.5-0.7 | **Moderate-high** | Reduce filters by 15-20% |
| 0.3-0.5 | **Moderate** | Reduce filters by 10-15% (optional) |
| < 0.3 | **Low redundancy** | Architecture is well-sized |

**Example**: `redundancy_score: 0.62` → Model has moderate-high redundancy, consider reducing encoder filters by 15-20%.

### Mean Correlation

**What it measures**: Average pairwise correlation between features

| Score | Meaning | Interpretation |
|-------|---------|---------------|
| > 0.6 | **Highly correlated** | Features are redundant, reduce filter count |
| 0.3-0.6 | **Moderately correlated** | Acceptable feature diversity |
| < 0.3 | **Diverse features** | Good feature learning |

### Trace

**What it measures**: Sum of eigenvalues (total variance)

- **Higher trace** = Model captures more variance
- **Lower trace** = Features are more compressed
- Use trace to normalize redundancy (already done by default)

### Condition Number

**What it measures**: Ratio of largest to smallest eigenvalue

| Score | Meaning | Action |
|-------|---------|--------|
| > 1e6 | **Numerical issues** | Consider adding regularization |
| 1e3-1e6 | **Acceptable** | Normal range |
| < 1e3 | **Well-conditioned** | No issues |

---

## Troubleshooting

### Zero NAS Metrics

**Problem**: `redundancy_score: 0.0, trace: 0.0, condition_number: 1.0`

**Causes**:
1. Monitoring single-channel output (segmentation mask)
2. No layer selectors specified

**Solution**:
```bash
# Add layer selectors to monitor internal activations
python src/train.py --model nano_u --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,bottleneck"
```

### Layer Not Found Error

**Problem**: `ValueError: Layer 'encoder_conv_0' not found in model`

**Solution**: List available layers
```python
import tensorflow as tf
model = tf.keras.models.load_model('models/nano_u.keras')
print([layer.name for layer in model.layers])
```

Then use correct layer names:
```bash
python src/train.py --model nano_u --enable-nas \
    --nas-layers "actual_layer_name_1,actual_layer_name_2"
```

### High Training Overhead

**Problem**: NAS monitoring significantly slows training

**Solution**: Reduce monitoring frequency
```bash
python src/train.py --model nano_u --enable-nas \
    --nas-log-freq epoch \
    --nas-batch-freq 10  # Only log every 10 epochs
```

### GPU Out of Memory

**Problem**: Training crashes with OOM error when NAS enabled

**Causes**:
- Monitoring too many layers
- Large activation tensors

**Solutions**:
```bash
# 1. Reduce number of monitored layers
python src/train.py --model nano_u --enable-nas \
    --nas-layers "bottleneck"  # Monitor only bottleneck

# 2. Reduce batch size
python src/train.py --model nano_u --enable-nas \
    --batch-size 4  # Smaller batch

# 3. Use gradient checkpointing (if available in model)
```

### CSV File Not Created

**Problem**: Training completes but no CSV file

**Causes**:
1. `save_history=False` in callback
2. Directory doesn't exist
3. Permission issues

**Solution**: Check logs and specify full path
```bash
python src/train.py --model nano_u --enable-nas \
    --nas-csv-path /full/path/to/metrics.csv
```

---

## Advanced Usage

### Custom Layer Monitoring Strategy

For U-Net architectures:

```bash
# Monitor encoder + bottleneck (skip decoder)
python src/train.py --model nano_u --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,encoder_conv_2,bottleneck"

# Monitor decoder only
python src/train.py --model nano_u --enable-nas \
    --nas-layers "decoder_conv_0,decoder_conv_1"

# Monitor everything with regex
python src/train.py --model nano_u --enable-nas \
    --nas-layers "/.*conv.*/"
```

### Automated Architecture Search

```bash
# Step 1: Train with NAS monitoring
python src/train.py --model nano_u --epochs 50 --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,bottleneck"

# Step 2: Analyze metrics
python src/plot_nas_metrics.py --csv logs/nas/nano_u_nas_metrics.csv

# Step 3: Interpret recommendations
# If redundancy > 0.5, edit config/config.yaml:
#   nano_u:
#     filters: [12, 24, 48]  # Reduced from [16, 32, 64]
#     bottleneck: 48         # Reduced from 64

# Step 4: Retrain with optimized architecture
python src/train.py --model nano_u --epochs 50
```

### Export Metrics to TensorBoard

NAS metrics are automatically logged to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir logs/nas

# Open browser to http://localhost:6006
```

---

## Best Practices

### 1. Always Monitor Internal Layers
```bash
# ❌ BAD: May return zeros
python src/train.py --model nano_u --enable-nas

# ✅ GOOD: Monitor internal conv layers
python src/train.py --model nano_u --enable-nas \
    --nas-layers "encoder_conv_0,encoder_conv_1,bottleneck"
```

### 2. Start with Conservative Reductions
```bash
# If redundancy = 0.6, don't immediately cut filters in half
# Reduce by 15-20% first, retrain, and re-evaluate
```

### 3. Monitor Both Teacher and Student
```bash
python scripts/tf_pipeline.py pipeline \
    --enable-nas-for-teacher \
    --enable-nas-for-student
```

### 4. Use Meaningful CSV Names
```bash
python src/train.py --model nano_u --enable-nas \
    --nas-csv-path results/nano_u_baseline_nas.csv

# After architecture changes
python src/train.py --model nano_u --enable-nas \
    --nas-csv-path results/nano_u_optimized_nas.csv
```

### 5. Compare Before and After
```bash
# Generate comparison plot
python src/plot_nas_metrics.py \
    --compare results/nano_u_baseline_nas.csv results/nano_u_optimized_nas.csv \
    --model-names "Baseline" "Optimized"
```

---

## References

- **Config**: [`config/config.yaml`](../config/config.yaml)
- **Training Script**: [`src/train.py`](../src/train.py)
- **NAS Implementation**: [`src/nas_covariance.py`](../src/nas_covariance.py)
- **Plotting Tool**: [`src/plot_nas_metrics.py`](../src/plot_nas_metrics.py)
- **Pipeline Script**: [`scripts/tf_pipeline.py`](../scripts/tf_pipeline.py)

---

**Last Updated**: 2026-01-19  
**Version**: 1.0
