# Nano-U

> Ultra-compact terrain segmentation for bare-metal microcontrollers, deployed via Rust.

**Nano-U** is designed specifically for agricultural robotics in unstructured environments where dense vegetation, domain shift, and viewpoint distortion present significant challenges. Unlike state-of-the-art models that require GPU inference with GBs of RAM, every architectural decision in Nano-U is motivated by the extreme constraints of commodity hardware — a mass-market SoC costing under $10, with no ML accelerator and a strict 320 KB DRAM ceiling.

The model is trained via **Quantization-Aware Distillation (QAD)** and deployed through a [fork of MicroFlow](https://github.com/federico-pizz/microflow-rs) extended with the operators required by Nano-U's architecture (`MaxPool2D`, nearest-neighbor upsampling, and `Sigmoid`), which are absent from the original engine.

---

## Performance Highlights

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
Nano-U relies on a strictly sequential encoder-decoder design to stay within the 320 KB DRAM budget. Skip connections are omitted entirely: retaining encoder feature maps until the decoder consumes them would exceed the memory budget at the target resolution.
- **Depthwise Separable Convolutions:** Used throughout (K=3) to achieve ~8-9× parameter and compute reduction relative to standard convolutions.
- **Compound Scaling:** Spatial dimensions and channel widths ([4, 8, 16]) are co-designed so the largest intermediate tensor is 30×40×4 = 19.2 KB, bounding peak arena occupancy within 320 KB.
- **Nearest-Neighbor Upsampling:** Replaces bilinear interpolation in the decoder — requires no multiplications and is supported in the extended MicroFlow fork.
- **Asymmetric Pooling:** A 3×2 MaxPool at Stage 3 reduces 15×20 cleanly to 5×10, avoiding the misalignment that a standard 2×2 pool would introduce.

### 2. Quantization-Aware Distillation (QAD)
A network of fewer than 3,400 parameters cannot generalize from hard binary labels alone. QAD addresses this by fusing two training objectives into a single pass, rather than applying them sequentially:
- **Knowledge Distillation:** Soft targets from BU-Net (12.85M parameters, trained to convergence) guide the student with temperature T=4.0 and α=0.5, exposing gradient signal on ambiguous boundary pixels that hard labels would suppress.
- **Quantization-Aware Training (QAT):** Fake-quantization nodes are injected from the first epoch, allowing the optimizer to simultaneously minimize the distillation loss and quantization error. Fusing QAT with distillation from the start prevents the student from converging to float32 weights that are subsequently perturbed by integer conversion.

### 3. Rust Deployment via `microflow-rs`

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/20/Rustacean-orig-noshadow.svg" alt="Rust Ferris logo" width="225" style="vertical-align: middle; margin-right: 20px;" />
  <a href="https://github.com/federico-pizz/microflow-rs">
    <img src="https://raw.githubusercontent.com/federico-pizz/microflow-rs/main/assets/microflow-logo.png" alt="microflow-rs logo" width="250" style="vertical-align: middle;" />
  </a>
</p>

Inference runs on bare-metal Xtensa LX7 using [our fork of MicroFlow](https://github.com/federico-pizz/microflow-rs), which extends the original engine with three operators required by Nano-U: `MaxPool2D`, nearest-neighbor `UpSampling2D`, and `Sigmoid` activation.
- **Compile-Time Graph Evaluation:** MicroFlow resolves the full operator graph at compile time via a Rust procedural macro. All tensor buffers are placed in static DRAM at link time — no heap allocator, no dynamic dispatch, no interpreter overhead.
- **`no_std` Environment:** The wireless stack is never linked. PSRAM is initialized exclusively for input image storage. Only the CPU, internal DRAM, and PSRAM are active during inference, yielding a measured board-level power draw of 500 mW and 422 mJ per inference — approximately 25× lower than comparable WiFi-enabled ESP32 deployments.

### 4. TinyAgri Dataset
To benchmark cross-domain generalization without camera-induced domain shift, we collected and publicly release **TinyAgri** — a terrain segmentation dataset captured directly via the onboard camera of an ESP32-CAM module in agricultural fields. Nano-U is evaluated on TinyAgri without any fine-tuning, achieving 66.7% mIoU under significant domain shift from the Botanic Garden training distribution.

---

## Project Structure

```text
Nano-U/
├── config/                  # Global and experiment-level configuration
│   ├── config.yaml          # Dataset paths, model shapes, global paths
│   ├── botanic_garden_config.yaml # Botanic Garden specific config
│   └── tinyagri_config.yaml       # TinyAgri specific config
├── src/                     # Core library code
│   ├── models/              # Model builders: Nano-U student and BU-Net teacher
│   ├── utils/               # QAT wrappers, config loaders, and metrics
│   ├── data.py              # tf.data.Dataset pipelines and augmentation
│   ├── evaluate.py          # Model visualization and evaluation tools
│   ├── nas.py               # Architecture monitoring and redundancy metrics (SVD profiling)
│   ├── pipeline.py          # Training orchestrators
│   ├── quantize_model.py    # Calibration and INT8 TFLite export
│   └── train.py             # Training loops with QAT and distillation support
├── scripts/                 # CLI entry points
│   ├── train_standard.py    # Standard training and export
│   ├── train_distillation.py# QAD: student-teacher distillation pipeline
│   └── stack_analyzer.py    # On-device stack and energy profiling
├── results/                 # Experiment logs, metrics, and visualization plots
├── tests/                   # Pytest suite
└── esp_flash/               # ESP32-S3 firmware (Rust `no_std`) and inference
```

---

## Quick Start

### 1. Installation

Requires Python 3.12+.

```bash
python -m venv nano_u_venv
source nano_u_venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a Quick Verification Test

Train Nano-U for 2 epochs to confirm the environment, datasets, and pipelines are configured correctly:

```bash
python scripts/train_standard.py nano_u --config config/botanic_garden_config.yaml
```

---

## Workflows and CLI

### Standard Training

Train models independently without Knowledge Distillation. You must explicitly specify whether to train the teacher (`bu_net`) or the student (`nano_u`).

```bash
# Train BU-Net using the default configuration
python scripts/train_standard.py bu_net

# Train Nano-U on a specific dataset configuration
python scripts/train_standard.py nano_u --config config/botanic_garden_config.yaml
```

### Quantization-Aware Distillation (QAD)

Trains both models in sequence: trains the BU-Net teacher to convergence, then trains the Nano-U student with fused distillation and INT8 QAT:

```bash
# Uses the default configuration
python scripts/train_distillation.py

# Uses a specific dataset configuration
python scripts/train_distillation.py --config config/tinyagri_config.yaml
```

---

## Inference and Export

INT8 export is triggered automatically after training. To manually export a `.keras` model to a calibrated INT8 `.tflite` file:

```bash
python src/quantize_model.py path/to/model.keras path/to/model.tflite
```

Calibration uses the validation split to compute INT8 scale factors and zero-points, matching the procedure described in the paper.

### MCU Deployment

See the `esp_flash/` directory for the Rust `no_std` firmware and its own `README.md`. The deployment uses our [MicroFlow fork](https://github.com/federico-pizz/microflow-rs). The model is loaded at compile time:

```rust
#[model("models/nano_u.tflite")]
struct UNet;

// Inference call
let mask = UNet::predict_quantized(input);
```

All tensor buffers fit within the 320 KB internal DRAM. No inference data is routed through PSRAM.

---

## Hardware Profiling

Peak DRAM consumption is measured via stack painting: the stack region is pre-filled with a known byte pattern (`0xAA`) before inference, and the high-water mark is recovered afterward by scanning for the first overwritten address. This yields a direct hardware measurement independent of static analysis estimates, inclusive of all MicroFlow tensor allocations.

```bash
# Requires ESP32-S3 connected via USB
python scripts/stack_analyzer.py
```

Results and plots are saved to the `results/` directory.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Future Work

- **Multi-core inference:** MicroFlow currently uses a single core; distributing inference across both Xtensa LX7 cores is the most direct path to reducing the 844 ms latency.
- **GPIO-isolated energy measurement:** Current figures are board-level and inclusive of the LDO, camera module, and USB-UART bridge. SoC-level isolation via GPIO switching would yield a more precise per-inference power figure.
- **On-device domain adaptation:** Deployment-time fine-tuning using the second core to adapt to new terrain environments without re-training from scratch.
- **Multi-class extension:** Expanding the output head to support soil, crop, and weed classification beyond binary traversability.