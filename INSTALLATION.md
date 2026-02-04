# Installation and Usage Guide

## System Requirements

### Software Dependencies

- **Python**: 3.12+ (tested with 3.12.12)
- **Operating System**: Linux (primary), macOS, Windows with WSL
- **GPU**: NVIDIA GPU with CUDA 13.1+ (optional but recommended)
- **Memory**: 8GB+ RAM, 16GB+ recommended for training

### Hardware Requirements

#### Development Environment
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: NVIDIA RTX series or equivalent (for training acceleration)
- **Storage**: 10GB+ available space
- **Network**: Internet connection for package installation

#### Target Deployment (ESP32-S3)
- **Microcontroller**: ESP32-S3 with 520KB SRAM, 8MB PSRAM
- **Flash**: 4MB+ for model storage
- **Power**: <1W target consumption
- **Peripherals**: Camera module for image input

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv-nano-u
source .venv-nano-u/bin/activate  # Linux/macOS
# .venv-nano-u\Scripts\activate  # Windows
```

### 3. Install Dependencies

#### Core Dependencies
```bash
# Essential packages
pip install tensorflow>=2.21 numpy opencv-python PyYAML

# Development dependencies (optional)
pip install pytest jupyter matplotlib pandas
```

#### GPU Support (Optional)
```bash
# Verify CUDA installation
nvidia-smi

# TensorFlow should automatically detect GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### 4. Verify Installation

```bash
# Test imports
python -c "
import tensorflow as tf
import numpy as np
import cv2
import yaml
print('TensorFlow version:', tf.__version__)
print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)
"
```

### 5. ESP32 Development Setup (Optional)

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Add ESP32 target
rustup target add xtensa-esp32s3-none-elf

# Install espflash
cargo install espflash
```

## Project Structure

```
Nano-U/
├── README.md                    # Project overview
├── RESEARCH_PAPER.md           # Detailed methodology
├── API_REFERENCE.md            # Code documentation
├── INSTALLATION.md             # This file
├── DEVELOPMENT.md              # Research roadmap
├── config/
│   └── config.yaml            # Training configuration
├── src/
│   ├── train.py               # Main training script
│   ├── models/
│   │   ├── Nano_U/           # Student model
│   │   └── BU_Net/           # Teacher model
│   ├── utils/                # Utilities (data, metrics, config)
│   ├── nas_covariance.py     # NAS monitoring
│   ├── quantize.py           # Model quantization
│   └── evaluate.py           # Model evaluation
├── scripts/
│   ├── run_experiments.py    # Experiment runner
│   └── infer_demo.py         # Inference demo
├── esp_flash/               # ESP32-S3 deployment
│   ├── Cargo.toml
│   └── src/
├── tests/                   # Unit tests
├── logs/                    # Training logs
└── models/                 # Trained models
```

## Quick Start

### 1. Prepare Dataset

The project expects a TinyAgri-style dataset structure:

```
data/
├── TinyAgri/
│   ├── Tomatoes/
│   │   ├── scene1/          # RGB images (.png)
│   │   └── scene2/
│   └── Crops/
│       ├── scene1/
│       └── scene2/
└── masks/
    ├── Tomatoes/
    │   ├── scene1/          # Binary masks (.png)
    │   └── scene2/
    └── Crops/
        ├── scene1/
        └── scene2/
```

Process the dataset:
```bash
python src/prepare_data.py
```

This creates `data/processed_data/` with train/val/test splits.

### 2. Train Teacher Model (Optional)

```bash
# Train BU_Net teacher model
python src/train.py --model bu_net --epochs 100
```

### 3. Train Student Model

```bash
# Basic student training
python src/train.py --model nano_u --epochs 100

# With knowledge distillation (requires teacher model)
python src/train.py --model nano_u --distill --teacher-weights models/bu_net.keras

# With NAS monitoring
python src/train.py \
  --model nano_u \
  --distill \
  --teacher-weights models/bu_net.keras \
  --enable-nas \
  --nas-layers "encoder_conv_0,encoder_conv_1,bottleneck"
```

### 4. Quantize for Deployment

```bash
# Convert to INT8 TensorFlow Lite
python src/quantize.py --model-name nano_u --output models/nano_u_int8.tflite
```

### 5. Evaluate Performance

```bash
# Evaluate model accuracy
python src/evaluate.py --model-name nano_u --out results/metrics.json

# Run inference demo
python scripts/infer_demo.py --model models/nano_u.keras --image test_image.png
```

## Advanced Usage

### Hyperparameter Experimentation

```bash
# Run systematic hyperparameter search
python scripts/run_experiments.py --phase 4.1 --output results/phase_4_1/

# Resume interrupted experiments
python scripts/run_experiments.py --phase 4.1 --resume --checkpoint results/checkpoint.json
```

### Custom Training Configuration

Edit `config/config.yaml`:

```yaml
training:
  nano_u:
    epochs: 200              # Extended training
    batch_size: 16           # Larger batches
    learning_rate: 5e-5      # Lower learning rate
    
    distillation:
      alpha: 0.2             # More teacher influence
      temperature: 3.0       # Different temperature

nas:
  enabled: true
  layer_selectors: ["encoder_conv_0", "encoder_conv_1", "bottleneck"]
  log_freq: "epoch"
```

### NAS Monitoring Analysis

```bash
# Generate NAS visualization
python src/plot_nas_metrics.py --csv logs/nas/nano_u_nas_metrics.csv

# Compare multiple models
python src/plot_nas_metrics.py \
  --compare logs/nas/nano_u_nas_metrics.csv logs/nas/bu_net_nas_metrics.csv \
  --model-names "Nano_U" "BU_Net"
```

### ESP32-S3 Deployment

```bash
# Build ESP32 firmware
cd esp_flash
cargo build --release --bin main

# Flash to device
espflash flash --monitor target/xtensa-esp32s3-none-elf/release/main

# Run analysis tools
cargo run --bin analysis
./run_analyzer.sh
```

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_nas_callback.py -v
pytest tests/test_nas_covariance.py -v
pytest tests/test_tf_pipeline.py -v
```

### Validate Installation

```bash
# Quick validation script
python -c "
from src.models.Nano_U.model_tf import build_nano_u
from src.models.BU_Net.model_tf import build_bu_net

# Test model building
nano_u = build_nano_u()
bu_net = build_bu_net()

print(f'Nano_U parameters: {nano_u.count_params():,}')
print(f'BU_Net parameters: {bu_net.count_params():,}')
print('Models built successfully!')
"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root directory, or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Nano-U"
```

#### 2. GPU Memory Issues
```bash
ResourceExhaustedError: OOM when allocating tensor
```
**Solutions:**
- Reduce batch size in config: `batch_size: 4`
- Enable memory growth (automatically handled in `src/train.py`)
- Use CPU: `export CUDA_VISIBLE_DEVICES=""`

#### 3. Model Instantiation Error
```bash
TypeError: InternalError.__init__() missing 2 required positional arguments
```
**Status**: Known issue under investigation
**Workaround**: Use TensorFlow 2.17 or functional API implementation

#### 4. Dataset Not Found
```bash
FileNotFoundError: No training data found
```
**Solution**: 
- Verify dataset structure matches expected format
- Run data preparation: `python src/prepare_data.py`
- Check paths in `config/config.yaml`

#### 5. NAS Negative Metrics
```
condition_number: -55814612.0
```
**Status**: Numerical instability issue
**Workaround**: Disable NAS monitoring temporarily

### Environment Debugging

```bash
# System information
python -c "
import tensorflow as tf
import numpy as np
import sys
import platform

print('Python:', sys.version)
print('Platform:', platform.platform())
print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('GPU Available:', tf.test.is_gpu_available())
if tf.config.list_physical_devices('GPU'):
    print('GPU Devices:', tf.config.list_physical_devices('GPU'))
"

# Check CUDA setup
nvidia-smi
nvcc --version
```

### Performance Optimization

#### Training Acceleration
- Use mixed precision training (experimental):
  ```python
  from tensorflow.keras.mixed_precision import Policy
  policy = Policy('mixed_float16')
  tf.keras.mixed_precision.set_global_policy(policy)
  ```

- Optimize data pipeline:
  ```python
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  dataset = dataset.cache()
  ```

#### Memory Management
- Monitor GPU memory usage:
  ```python
  import tensorflow as tf
  print(tf.config.experimental.get_memory_info('GPU:0'))
  ```

- Clear memory between runs:
  ```python
  tf.keras.backend.clear_session()
  import gc; gc.collect()
  ```

## Development Environment

### Jupyter Notebook Setup

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook

# Open training workflow
# Navigate to notebooks/training_workflow.ipynb
```

### IDE Configuration

#### VS Code
1. Install Python extension
2. Configure Python interpreter: `.venv-nano-u/bin/python`
3. Install extensions: Python, Jupyter, TensorFlow snippets

#### PyCharm
1. Configure Python interpreter
2. Set source root: `src/`
3. Configure run configurations for scripts

### Code Formatting

```bash
# Install development tools
pip install black isort flake8

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Check style
flake8 src/ scripts/ tests/
```

## Contributing

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
python -m venv .venv-dev
source .venv-dev/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src/ --cov-report=html

# Generate test report
pytest tests/ --html=reports/test_report.html
```

---

**Support**: For installation issues, create an issue on GitHub with system details and error logs.
**Last Updated**: 2026-02-04