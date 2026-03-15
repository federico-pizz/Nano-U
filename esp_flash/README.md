# Nano-U: Rust Bare-Metal Inference (`esp_flash/`)

> Zero-allocation, `no_std` deterministic execution for the Nano-U model on the ESP32-S3 using `microflow-rs`.

This directory contains the firmware required to run the trained INT8 Nano-U models directly on the ESP32-S3-CAM hardware. It leverages the Rust embedded ecosystem to achieve maximum efficiency within the strict memory constraints of commodity microcontrollers.

---

## Why Rust and `microflow-rs`?

Given the absence of an ML accelerator or hardware FPU on the ESP32-S3, and a hard ceiling of ~320 KB for DRAM, standard interpreter-based frameworks (like TFLite Micro) present challenges due to dynamic dispatch overhead and memory fragmentation from heap allocations.

Instead, we use [microflow-rs](https://github.com/microflow-rs/microflow-rs) (Note: Update link if different):
- **Compile-Time Graph Evaluation:** All tensor shapes and operations are known at compile time and encoded into the Rust type system. 
- **Zero Dynamic Allocation (no heap):** The compiled binary contains the entire inference graph as a sequence of statically allocated operations.
- **Deterministic Execution:** Eliminates memory faults and latency spikes associated with garbage collection or heap fragmentation.

### Hardware Efficiency Gains
- **Wireless Isolation:** The WiFi and Bluetooth stacks are never initialized in our `no_std` Rust build, saving ~5-15 mA compared to a standard C++ ESP-IDF deployment.
- **PSRAM Bypass:** We explicitly avoid initializing the 8 MB octal PSRAM, bypassing an additional ~10-20 mA required for refresh currents.

---

## Memory Layout and Inference Strategy

### The 320 KB DRAM Arena
All tensors—including the peak intermediate tensor of 19.2 KB (60x80x4) generated during the first encoder stage—live entirely in statically allocated DRAM. 
- **Stack Painting Profile:** Our profiling shows a total allocated region of 312.6 KB, with an actual high-water mark peak usage of **257 KB**.

### Instruction RAM (IRAM)
Currently, the inference code lives in flash memory and is fetched through the instruction cache. The remaining ~192 KB of internal SRAM is designated as IRAM. 
*Future Optimization:* Pinning hot loops directly to IRAM using the `#[ram]` attribute is projected to reduce latency by a further 10-20% through single-cycle fetches.

### The Quantization Pipeline
- **Build-Time Extraction:** Input scales and zero-points are extracted directly from the `.tflite` model at build time via `build.rs`.
- **Pre-computed LUT:** Preprocessing uses a per-channel Look-Up Table (256 entries x 3 channels) computed once prior to inference. This avoids expensive floating-point operations per pixel on the soft-float CPU.
- **Thresholded Output:** The final INT8 output tensor is mapped directly to our background vs. traversable binary mask using the dequantized scale and zero-point.

---

## Profiling Tools: Stack Analyzer

This crate includes code specifically written to profile the ESP32-S3.

- **Stack Painting:** Before inference begins, the entire DRAM stack region is filled with a known byte pattern (`0xAA`). After inference, a bottom-up memory scan identifies the first modified byte to establish the precise high-water mark of RAM usage.
- **Energy Measurement Preparation:** The inference is designed to run in a continuous steady-state loop, allowing board-level power measurements (V/A) via a multimeter on the 5V USB rail.

---

## Build and Flash

*(Assuming the standard Rust ESP toolchain is installed)*

First, export the ESP-RS environment variables (path may vary depending on your installation):

```bash
source $HOME/export-esp.sh
```

This project contains two primary binaries:
1. `inference`: Runs the standard segmentation inference loop.
2. `analysis`: Runs the stack painting and energy profiling steady-state loop.

To build and flash a specific binary to the ESP32-S3 and monitor the serial output:

```bash
# Run the standard inference binary
cargo run --release --bin inference

# Run the stack and energy profiling binary
cargo run --release --bin analysis
```