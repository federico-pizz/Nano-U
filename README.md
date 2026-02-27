# Nano-U 🔬🌱

> Ultra-low-power CNN for real-time semantic segmentation on energy-constrained microcontrollers (e.g. ESP32-S3). 

**Nano-U** is designed specifically for agricultural robotics and edge-AI scenarios where extreme parameter efficiency, low latency, and power consumption are critical. Through Evolutionary Neural Architecture Search (NAS), Knowledge Distillation, and robust INT8 Quantization, it delivers accurate semantic segmentation within a budget of less than 3,000 parameters.

---

## Key Features 🚀

- **Evolutionary NAS**: Custom evolutionary search optimizing for both high binary IoU and minimal structural redundancy via SVD profiling. 
- **Knowledge Distillation**: Train a high-capacity Teacher (BU-Net) and distill its visual priors into the tiny Nano-U Student model.
- **Microflow / Bare-Metal Friendly**: Strict adherence to operations supported by target MCUs. Full INT8 TFLite export ensures compatibility with constraint environments.
- **Config-Driven Pipelines**: Easily define hyperparameter sweeps, architecture specifications, and training regimes through structured YAML files.

---

## Project Structure 📁

```text
Nano-U/
├── config/                  # Global and experiment-level configuration
│   ├── config.yaml          # Dataset paths, model shapes, global paths
│   └── experiments.yaml     # Pre-defined experiments (standard, distillation, nas, quick_test)
├── src/                     # Core library code
│   ├── models/              # Model builders, layer primitives, and architecture definitions
│   ├── data.py              # tf.data.Dataset pipelines and augmentation
│   ├── evaluate.py          # Model visualization and evaluation tools
│   ├── nas.py               # Evolutionary Search and Redundancy metrics
│   ├── pipeline.py          # Training and search orchestrators
│   ├── quantize_model.py    # Representative dataset calibration and INT8 conversion
│   └── train.py             # Training loops (Standard & Distillation)
├── scripts/                 # CLI Entry Points
│   ├── train_standard.py    # Train and export models (e.g., standard training, quick_test)
│   └── train_distillation.py# Student-Teacher knowledge distillation pipeline
├── tests/                   # Pytest suite
└── esp_flash/               # Microcontroller firmware, memory safety, and native inference
```

---

## Quick Start 🛠

### 1. Installation

Ensure you have Python 3.12+ and set up your virtual environment:

```bash
python -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r requirements.txt
```

### 2. Run a Quick Verification Test

Train Nano-U for 2 epochs to ensure the environment, datasets, and pipelines are configured correctly:

```bash
python scripts/train_standard.py --experiment quick_test
```

---

## Workflows and CLI 🎛

### Standard Training

Train models from scratch or load specific experiments defined in `config/experiments.yaml`.

```bash
# Train using the default configuration
python scripts/train_standard.py

# Specify an experiment and target model
python scripts/train_standard.py --experiment standard_training --model nano_u
```

### Knowledge Distillation

Train the BU-Net teacher, transfer knowledge via soft-targets, and train the Nano-U student.

```bash
python scripts/train_distillation.py --experiment distillation_nas
```

---

## Inference and Export 📦

Post-training quantization is automatically triggered after successful training and NAS workflows. To manually quantize a `.keras` model to full INT8 precision utilizing the exact processed validation dataset for proper scale calibration:

```bash
python src/quantize_model.py path/to/model.keras path/to/model.tflite
```

### MCU Deployment
Check the `esp_flash/` directory for Rust code (`no_std`), memory optimization strategies, IRAM execution scripts, and stack analysis required for loading the `.tflite` directly to the ESP32-S3.

---

## Running Tests 🧪

To verify the integrity of model building, quantization, data pipelines, and evolutionary search components:

```bash
python -m pytest tests/ -v
```
