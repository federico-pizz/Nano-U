# esp_flash (Rust ESP32-S3 project)

This directory contains the Rust project targeting ESP32-S3 used for tiny inference and stack usage analysis. It was merged into the main repository and cleaned of nested git metadata and large build artifacts.

Quickstart
----------
- Ensure the ESP Rust toolchain is installed (see `rust-toolchain.toml`).
- Build the analysis binary:

```bash
cd esp_flash
cargo build --release --bin analysis
```

- Flash and capture UART log:

```bash
./run_analyzer.sh
```

Notes
-----
- Models were moved to the repository root `models/` directory.
- Large ephemeral artifacts (target/, venvs) were removed from this directory during merge.
