# Nano-U

> Ultra-low-power CNN for real-time semantic segmentation on energy-constrained microcontrollers (e.g., commodity ESP32-S3-CAM).

**Nano-U** is designed specifically for agricultural robotics in unstructured environments where dense vegetation, domain shift, and viewpoint distortion present significant challenges. Unlike state-of-the-art models that require GPU inference with GBs of RAM, every architectural decision in Nano-U is motivated by the extreme constraints of commodity hardware (a mass-market SoC costing under $10, with no ML accelerator or DSP extensions, and a strict 320 KB DRAM ceiling).

---

## Performance Highlights

Nano-U achieves impressive accuracy within severe hardware constraints for binary terrain segmentation (traversable path vs. background).

| Metric | Botanic Garden Dataset | Tiny Agri Dataset (Zero-Shot) |
| :--- | :--- | :--- |
| **mIoU** | 80.3% | 66.7% |
| **Parameters** | 3,357 | 3,357 |
| **Model Footprint** | 34 KB (.tflite) | 34 KB (.tflite) |
| **Peak DRAM Usage** | 257 KB | 257 KB |
| **Total Stack** | 313 KB | 313 KB |
| **Latency** | 844 ms / inference | 844 ms / inference |
| **Energy Consumption** | 422 mJ / inference (board-level) | 422 mJ / inference (board-level) |

---

## Key Contributions & Methodology

### 1. Constraint-Driven Architecture
Nano-U relies on a strictly sequential encoder-decoder design to stay within the 300 KB DRAM budget.
- **Depthwise Separable Convolutions:** Used throughout (K=3) to achieve ~8-9x parameter and compute reduction on the soft-float Xtensa LX7 CPU.
- **Compound Scaling:** Manually adapted width and resolution stages to ensure the peak tensor (19.2 KB) remains within limits.
- **Nearest-Neighbor Upsampling:** Replaces expensive bilinear interpolation with zero-arithmetic memory copying, natively supported in INT8 by microflow-rs.

### 2. Quantization-Aware Distillation (QAD)
To ensure the tiny 3,357-parameter student can generalize:
- **Knowledge Distillation (KD):** Uses soft targets from a strong BU-Net teacher (alpha=0.5, T=4.0) to provide essential "dark knowledge".
- **Quantization-Aware Training (QAT):** Fake-quant nodes injected during training simulate INT8 rounding, allowing the optimizer to find robust weights before deployment, minimizing accuracy degradation compared to Post-Training Quantization (PTQ).

### 3. Rust Deployment via `microflow-rs`

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/20/Rustacean-orig-noshadow.svg" alt="Rust Ferris logo" width="225" style="vertical-align: middle; margin-right: 20px;" />
  <a href="https://github.com/federico-pizz/microflow-rs">
    <img src="https://raw.githubusercontent.com/federico-pizz/microflow-rs/main/assets/microflow-logo.png" alt="microflow-rs logo" width="250" style="vertical-align: middle;" />
  </a>
</p>

The model is deployed using `microflow-rs`, chosen specifically because it requires ZERO dynamic allocation.
- **Compile-Time Graph Evaluation:** All tensor sizes are known at compile time, eliminating the overhead of TFLite Micro's interpreter and dynamic dispatch.
- **`no_std` Environment:** Ensures memory safety without GC and deterministic execution. Crucially, the WiFi stack and PSRAM are never initialized, saving significant milliamp draw compared to standard C++ IDF deployments.

### 4. Tiny Agri Dataset
To eliminate domain shift from camera optics, we gathered and released **Tiny Agri**, a custom agricultural field dataset collected directly using the ESP32-CAM onboard sensor.

---

## Future Work

We are actively exploring improvements to the pipeline, including:
- **IRAM Optimization:** Pinning hot loops to Instruction RAM (`#[ram]`) for an estimated single-cycle fetch latency reduction of 10-20%.
- **Temporal Depthwise Convolutions:** To improve video-rate consistency.
- **SVD-Based Auto-Pruning:** Integrating auto-pruning during QAT to push the parameter count below 3,000.
- **Hardware-In-The-Loop NAS (HIM-NAS):** Feeding live energy measurements from our onboard profiler directly into the NAS fitness function.
- **Continual Learning Support:** Enabling on-device fine-tuning using the second core.
- **Multi-Class Extension:** Expanding the bottleneck to support soil, crop, and weed classification.

---

## Project Structure

```text
Nano-U/
├── config/                  # Global and experiment-level configuration
│   ├── config.yaml          # Dataset paths, model shapes, global paths
│   └── experiments.yaml     # Pre-defined experiments (standard, distillation, nas, quick_test)
├── src/                     # Core library code
│   ├── models/              # Model builders, layer primitives, and architecture definitions
│   ├── utils/               # QAT wrappers, Config loaders, and metrics
│   ├── data.py              # tf.data.Dataset pipelines and augmentation
│   ├── evaluate.py          # Model visualization and evaluation tools
│   ├── nas.py               # Evolutionary Search and Redundancy metrics
│   ├── pipeline.py          # Training and search orchestrators
│   ├── quantize_model.py    # Representative dataset calibration and INT8 conversion
│   └── train.py             # Training loops with QAT & Distillation support
├── scripts/                 # CLI Entry Points
│   ├── train_standard.py    # Train and export models
│   ├── train_distillation.py# Student-Teacher distillation pipeline
│   └── stack_analyzer.py    # On-device stack & energy profiling
├── results/                 # Experiment logs, metrics, and visualization plots
├── tests/                   # Pytest suite
└── esp_flash/               # ESP32-S3 Firmware (Rust `no_std`) & Inference
```

---

## Quick Start

### 1. Installation

Ensure you have Python 3.12+ and set up your virtual environment:

```bash
python -m venv nano_u_venv  
source nano_u_venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a Quick Verification Test

Train Nano-U for 2 epochs to ensure the environment, datasets, and pipelines are configured correctly:

```bash
python scripts/train_standard.py --experiment quick_test
```

---

## Workflows and CLI

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

## Inference and Export

Post-training quantization is automatically triggered after successful training and NAS workflows.

### Quantization-Aware Training (QAT)
For `nano_u` models, QAT is enabled by default in `config/config.yaml`. This ensures that the model is aware of the INT8 quantization constraints during training, significantly improving on-device accuracy compared to post-training quantization alone.

### Manual Export
To manually quantize a `.keras` model to full INT8 precision utilizing the exact processed validation dataset for proper scale calibration:

```bash
python src/quantize_model.py path/to/model.keras path/to/model.tflite
```

### MCU Deployment
Check the `esp_flash/` directory for Rust code (`no_std`), memory optimization strategies, IRAM execution scripts, and stack analysis required for loading the `.tflite` directly to the ESP32-S3. **For detailed information about the Rust part, please refer to the `README.md` inside the `esp_flash/` directory.**

This crate contains the firmware built on the Rust embedded (`no_std`) ecosystem, running a completely statically allocated inference graph. The binary executes entirely from flash and statically allocated DRAM (~312 KB high-water mark), resulting in strongly deterministic runtimes and zero runtime memory fragmentation.

---

## Performance & Energy Analysis

The project includes a specialized tool to profile the model directly on the ESP32-S3 hardware. It measures memory overhead, execution speed, and estimated power consumption.

```bash
# Run on-device analysis (requires ESP32-S3 connected via USB)
python scripts/stack_analyzer.py
```

**Metrics Provided:**
- **Stack Peak Usage**: The maximum memory consumed by the TFLite Micro interpreter.
- **Inference Latency**: Average time (ms) to process a single 80x60 frame.
- **Energy per Inference**: Calculated in milliJoules (mJ) based on measured current draw.

Results and visualization plots (e.g., `stack_usage.png`) are automatically saved to the `results/` directory.

---

## Running Tests

To verify the integrity of model building, quantization, data pipelines, and evolutionary search components:

```bash
python -m pytest tests/ -v
```
