# ESP32-S3 Embedded Inference (esp_flash)

Rust `no_std` project for running Nano-U inference on ESP32-S3 with stack usage analysis.

---

## 🎯 Features

- **Microflow Inference** – TFLite model execution via [`microflow-rs`](https://github.com/matteocarnelos/microflow-rs)
- **Stack Analysis** – Runtime stack watermarking to measure peak usage
- **240MHz Operation** – Maximum CPU clock for inference speed
- **Static Memory Allocation** – Buffers in `.bss` to avoid stack overflow

---

## 📋 Prerequisites

1. **ESP Rust Toolchain**
   ```bash
   # Install espup
   cargo install espup
   espup install
   
   # Source the environment
   source ~/export-esp.sh
   ```

2. **Python** (for analysis visualization)
   ```bash
   pip install matplotlib numpy
   ```

---

## 🚀 Quick Start

### Build & Flash
```bash
cd esp_flash

# Build analysis binary
cargo build --release --bin analysis

# Run full workflow (build → flash → capture → visualize)
./run_analyzer.sh
```

### Manual Flash
```bash
espflash flash target/xtensa-esp32s3-none-elf/release/analysis --monitor
```

---

## 📁 Structure

```
esp_flash/
├── src/
│   ├── bin/
│   │   ├── analysis.rs     # Stack analysis binary (50 inference runs)
│   │   └── main.rs         # Simple inference demo
│   └── lib.rs
├── models/                  # TFLite models (nano_u.tflite)
├── run_analyzer.sh          # Complete analysis workflow
├── run_inference.py         # UART capture helper
├── stack_analyzer.py        # Visualization generator
├── Cargo.toml
└── rust-toolchain.toml      # ESP Rust nightly config
```

---

## 🔬 Stack Analysis

The `analysis` binary uses **stack painting** to measure peak stack usage:

1. Fills unused stack with `0xAA` pattern before inference
2. Runs 50 inference iterations with real image data
3. Scans stack to find high-water mark
4. Reports `STACK_PEAK` and `STACK_TOTAL` via UART

### Output Example
```
Running Inference Iteration 1...
Inference done in 87 ms
STACK_PEAK:12456
STACK_TOTAL:32768
```

### Visualization
```bash
python stack_analyzer.py  # Generates stack_usage.png
```

---

## ⚙️ Configuration

### Cargo.toml Profiles
```toml
[profile.release]
opt-level = 3       # Maximum optimization
lto = 'fat'         # Link-time optimization
codegen-units = 1   # Better LLVM optimization
```

### Memory Configuration
Edit `.cargo/config.toml` to adjust stack size:
```toml
[env]
ESP_STACK_SIZE = "32768"  # 32KB stack
```

---

## 📖 Additional Documentation

- **[RUST_ESP32S3_NOSTD.md](RUST_ESP32S3_NOSTD.md)** – Detailed Rust `no_std` setup guide
- **[../README.md](../README.md)** – Main project documentation

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Stack overflow | for now just reducing the parameters of the model |
| Build fails | Run `source ~/export-esp.sh` first |
| No UART output | Check `/dev/ttyUSB0` permissions |
| Inference timeout | Watchdog is disabled; check model compatibility |

---

**Target**: ESP32-S3 @ 240MHz  
**Toolchain**: `esp` channel (nightly)  
**Framework**: `esp-hal` 1.0+
