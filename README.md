# Nano-U · Multicore

> **The dual-core inference variant of [Nano-U](../../tree/main).**
> Same 3,357-parameter model, same accuracy — but each heavy layer is split across both Xtensa LX7 cores for **~2× lower latency: 2.35 FPS on an ESP32-S3.**

[![Rust](https://img.shields.io/badge/rust-esp--rs-red.svg)](https://esp-rs.github.io/book/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

This branch is identical to [`main`](../../tree/main) except for the firmware inference path. Output is **bit-identical** to single-core, so **model, training, datasets, and accuracy are unchanged** — see the [`main` README](../../tree/main#nano-u) for all of that.

Everything below is the only thing that differs.

## What's different

Every inference binary brings the ESP32-S3 **APP core (core 1)** up into MicroFlow's `worker_loop` via the `start_dual_core!` macro (`firmware/src/lib.rs`), then MicroFlow splits each heavy conv/depthwise layer's output rows across both cores. The boot banner prints `WORKER_READY:1` once core 1 is polling.

This branch pairs with the `multicore` branch of the [MicroFlow fork](https://github.com/federico-pizz/microflow-rs) (`features = ["buffer-reuse", "multicore", "profiling"]`).

### Hardware Efficiency (ESP32-S3-CAM)

| Metric | Single-core (`main`) | **Multicore (this branch)** |
|:---|:---:|:---:|
| Inference latency | 830 ms (~1.2 FPS) | **425 ms (~2.35 FPS)** |
| Peak internal RAM | 281 KB | **147.5 KB** |
| Stack headroom | — | 130.7 KB |
| Power consumption | 470 mW | — |

Peak RAM measured via stack painting (hardware high-water mark) over 50 iterations. Latency measured with both Xtensa LX7 cores active.

## Build & run

The training/evaluation stack is unchanged — follow the [`main` README](../../tree/main#installation). For firmware build, flashing, and binary descriptions, see [`firmware/README.md`](firmware/README.md).

---

## License

Dual-licensed under the **MIT License** and the **Apache License 2.0** — use it under either, at your option. See [LICENSE](LICENSE).
