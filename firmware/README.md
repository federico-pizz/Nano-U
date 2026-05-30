# Nano-U Firmware

Bare-metal Rust inference for the ESP32-S3. No RTOS, no heap allocator — the full operator graph is resolved at compile time by [MicroFlow](https://github.com/federico-pizz/microflow-rs).

## Requirements

Install the ESP Rust toolchain, then source it before every build session:

```bash
# https://esp-rs.github.io/book/installation/index.html
source $HOME/export-esp.sh
```

## Binaries

| Binary | Description |
|:---|:---|
| `run` | Continuous inference loop (default `cargo run` target); prints per-frame summary |
| `inference` | One-shot benchmark over all packed test images; per-image timing, stats, and hex dump |
| `single_inference` | Single-image inference; streams the output mask over serial for host-side capture |
| `analysis` | Stack painting + power profiling for Nano-U |
| `analysis_person_detect` | Same profiling pipeline for the MobileNet-based person-detect baseline |

## Usage

```bash
# Run a binary (builds automatically)
cargo run --release --bin inference

# Target a specific dataset
MODELS_DIR=../models/TinyAgri \
TEST_IMG_DIR=../data/TinyAgri/test/img \
cargo run --release --bin inference

# Select a single image by index
TARGET_IMG_IDX=42 cargo run --release --bin single_inference
```

## Build-time configuration

The model and quantization parameters are embedded at compile time via `build.rs`. All defaults target BotanicGarden.

| Env var | Default | Description |
|:---|:---|:---|
| `MODELS_DIR` | `../models/BotanicGarden` | Directory containing `.tflite` and `_quant_params.json` |
| `TEST_IMG_DIR` | `../data/BotanicGarden/test/img` | PNG images packed into the binary |
| `MODEL_NAME` | `nano_u` | Stem of the model file |
| `TARGET_IMG_IDX` | `1` | Image index used by `single_inference` |

## Project structure

```
firmware/
├── src/
│   ├── bin/
│   │   ├── run.rs                    # Continuous inference loop (default target)
│   │   ├── inference.rs              # One-shot benchmark over all test images
│   │   ├── single_inference.rs       # Single-image + serial output
│   │   ├── analysis.rs               # Stack painting + power profiling
│   │   └── analysis_person_detect.rs # MobileNet baseline profiling
│   └── lib.rs                        # Shared helpers (quant constants, preprocess, stack utils)
├── models/                           # Copied here by build.rs at compile time
├── build.rs                          # Quantization param extraction + image packing
└── Cargo.toml
```
