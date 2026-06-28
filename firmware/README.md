# Nano-U Firmware

Bare-metal Rust inference for the ESP32-S3. No RTOS, no heap allocator — the full operator graph is resolved at compile time by [MicroFlow](https://github.com/federico-pizz/microflow-rs).

## Pipelines (branches)

The firmware ships in two parallel pipelines, one per branch, each paired with a
matching MicroFlow branch:

| Pipeline | nano-u branch | MicroFlow branch | features |
|:---|:---|:---|:---|
| Single-core | `main` | `buffer-reuse` | `["buffer-reuse"]` |
| **Multicore** (this branch) | `multicore` | `multicore` | `["buffer-reuse", "multicore", "profiling"]` |

On the multicore branch every inference binary brings the ESP32-S3 **APP core
(core 1)** up into MicroFlow's `worker_loop` via the `start_dual_core!` macro
(`src/lib.rs`), then MicroFlow splits each heavy conv/depthwise layer's output
rows across both cores. The boot banner prints `WORKER_READY:1` once core 1 is
polling. Output is bit-identical to single-core — only latency changes.

## Requirements

Install the ESP Rust toolchain, then source it before every build session:

```bash
# https://esp-rs.github.io/book/installation/index.html
source $HOME/export-esp.sh
```

## Binaries

| Binary | Description |
|:---|:---|
| `online` | **Live-camera** inference: captures OV2640 frames and prints a navigation decision per frame |
| `capture` | **Live-camera** frame dump: streams the raw + downscaled images over serial for pipeline validation (no inference) |
| `run` | Continuous inference loop over baked-in images (default `cargo run` target); prints per-frame summary |
| `inference` | One-shot benchmark over all packed test images; per-image timing, stats, and hex dump |
| `single_inference` | Single-image inference; streams the output mask over serial for host-side capture |
| `analysis` | Stack painting + power profiling for Nano-U (dual-core) |
| `analysis_prof` | Per-layer latency profile (dual-core) — op-type + microseconds per layer |
| `analysis_person_detect` | Same profiling pipeline for the MobileNet-based person-detect baseline (single-core) |

## Live camera (`online`)

The `online` binary is the real-time control loop on the **Goouuu ESP32-S3-CAM**
(OV2640). It replaces the baked-in image source with live capture: each frame is
DMA'd as QQVGA (160×120) RGB565 into PSRAM, box-downscaled 2×2 and INT8-quantized
into the 60×80 model input, run through the network, and reduced to a navigation
decision printed over serial:

```text
FRAME:7 infer=830ms cover L=0.62 C=0.81 R=0.40 -> GO steer=CENTER
```

```bash
MODELS_DIR=../models/TinyAgri cargo run --release --bin online
```

The camera pin map (XCLK=15, PCLK=13, VSYNC=6, HREF=7, SDA=4, SCL=5,
D0..D7=11,9,8,10,12,18,17,16; PWDN/RESET unwired → software reset) and the
OV2640 RGB565/QQVGA register table live in `src/camera.rs`; the pure decision
policy (host-testable) lives in `src/control.rs`. Capture uses PSRAM for the
frame buffer only — inference latency/throughput are unchanged.

### Validating the pipeline (`capture`)

To *see what the camera sees* and check each stage independently, the `capture`
binary runs the same bring-up as `online` but stops before inference and streams
the intermediate images over serial: `RAW` (160×120 RGB565 from the sensor) and
`DOWN` (60×80 RGB888 after the 2×2 box downscale — the pixels the quantizer
feeds the model). The host decoder saves each as a PNG:

```bash
# Terminal 1 — flash + run the streamer
MODELS_DIR=../models/TinyAgri cargo run --release --bin capture

# Terminal 2 — decode the serial stream into PNGs (or run instead of terminal 1
# with --no-reset against an already-running device)
python ../scripts/capture_view.py -n 5 --upscale 4 -o capture_out
```

`single_inference` covers the third stage (it streams the output mask), so the
three together validate the whole capture → downscale → infer chain. Flip
`SWAP_RGB565_BYTES` in `src/camera.rs` (and `--swap-rgb565` on the script) if
colours look wrong.

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

The model and quantization parameters are embedded at compile time via `build.rs`. Point the env vars below at any dataset's model directory; the defaults target BotanicGarden.

| Env var | Default | Description |
|:---|:---|:---|
| `MODELS_DIR` | `../models/BotanicGarden` | Directory containing `.tflite` and `_quant_params.json` |
| `TEST_IMG_DIR` | `../data/BotanicGarden/test/img` | PNG images packed into the binary |
| `MODEL_NAME` | `nano_u` | Stem of the model file |
| `TARGET_IMG_IDX` | `1` | Image index used by `single_inference` |

### Calibration parameters (required before first build)

`build.rs` embeds the model's INT8 quantization scales, zero-points, and input
normalization from a `<MODEL_NAME>_quant_params.json` file sitting next to the `.tflite`
in `MODELS_DIR`. This JSON is a **generated artifact and is not committed** — without it
`build.rs` panics. Regenerate it for whichever dataset you're building (run from the repo
root, inside the Python env), substituting your dataset name for `<dataset>`:

```bash
python -c "import json; from src.quantize_model import extract_quant_params; \
p = extract_quant_params('models/<dataset>/nano_u.tflite', 'config/config.yaml'); \
json.dump(p, open('models/<dataset>/nano_u_quant_params.json', 'w'), indent=2)"
```

For the models shipped in this repo, `<dataset>` is `BotanicGarden` or `TinyAgri`. The
full training pipeline (`scripts/train_model.py`) also writes this file automatically.

## Project structure

```
firmware/
├── src/
│   ├── bin/
│   │   ├── online.rs                 # Live-camera control loop (OV2640 → decision)
│   │   ├── capture.rs                # Live-camera frame dump for pipeline validation
│   │   ├── run.rs                    # Continuous inference loop (default target)
│   │   ├── inference.rs              # One-shot benchmark over all test images
│   │   ├── single_inference.rs       # Single-image + serial output
│   │   ├── analysis.rs               # Stack painting + power profiling (dual-core)
│   │   ├── analysis_prof.rs          # Per-layer latency profile (dual-core)
│   │   └── analysis_person_detect.rs # MobileNet baseline profiling
│   ├── camera.rs                     # OV2640 live-capture driver (hardware-only)
│   ├── control.rs                    # Pure navigation policy (host-testable)
│   └── lib.rs                        # Shared helpers (quant constants, preprocess, stack utils, start_dual_core!)
├── models/                           # Copied here by build.rs at compile time
├── build.rs                          # Quantization param extraction + image packing
└── Cargo.toml
```
