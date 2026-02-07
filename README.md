# Nano-U: Ultra-Low-Power CNN for Microcontroller Real-Time Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
![TensorFlow 2.21+](https://img.shields.io/badge/tensorflow-2.21+-blue)
![ESP32-S3](https://img.shields.io/badge/target-ESP32--S3-green)
![Research](https://img.shields.io/badge/status-research-orange)

> **Research Goal**: Real-time semantic segmentation for autonomous navigation on energy-constrained microcontrollers (ESP32-S3) with <100ms latency and <1W power consumption.

---

## 🔬 Overview

**Nano-U** investigates extreme CNN miniaturization for edge robotics:

| Feature | Description |
|---------|-------------|
| **Knowledge Distillation** | 180K → 41K parameters (77% reduction) |
| **Depthwise Separable Convs** | Optimized for MCU memory constraints |
| **INT8 Quantization** | ~10KB final model size |
| **Microflow Compatible** | Rust-based `no_std` inference engine |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
python -m venv .venv-tf && source .venv-tf/bin/activate
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
# Training → Distillation → Quantization → Benchmarking
python scripts/run_pipeline.py --full
```

### Individual Commands
```bash
# List available experiments
python scripts/run_pipeline.py --list

# Run specific experiment
python scripts/run_pipeline.py --experiment quick_test

# Evaluate model
python src/evaluate.py --model-name nano_u
```

---

## 📁 Project Structure

```
Nano-U/
├── src/                    # Python source code
│   ├── models/             # Model architectures (Nano-U, BU-Net)
│   ├── utils/              # Utilities (metrics, callbacks)
│   ├── train.py            # Training logic with distillation
│   ├── nas.py              # Neural Architecture Search
│   ├── evaluate.py         # Model evaluation
│   ├── quantize_model.py   # INT8 quantization
│   └── benchmarks.py       # Performance benchmarking
├── esp_flash/              # ESP32-S3 Rust inference (see esp_flash/README.md)
├── config/                 # YAML configuration files
│   ├── config.yaml         # Main training config
│   └── experiments.yaml    # Experiment definitions
├── scripts/                # Pipeline automation
├── models/                 # Saved models (.keras, .tflite)
├── data/                   # Training datasets
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks
```

---

## 📊 Results

| Metric | Teacher (BU-Net) | Student (Nano-U) | Reduction |
|--------|------------------|------------------|-----------|
| Parameters | 180K | 41K | - |
| Model Size | ~720KB | ~164KB | **77%** |
| Quantized | — | ~10KB | **98.6%** |

---

## 🛠️ Development

### Run Tests
```bash
pytest tests/ -v
```

### Configuration
Edit `config/config.yaml` for training parameters and `config/experiments.yaml` for experiment definitions.

---

## 📚 Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** – API and CLI documentation
- **[DEVELOPMENT.md](DEVELOPMENT.md)** – Development guide and roadmap
- **[esp_flash/README.md](esp_flash/README.md)** – ESP32-S3 deployment guide

---

## 📜 License

MIT License – see [LICENSE](LICENSE)

---

**Last Updated**: 2026-02-07  
**Status**: Active Research
