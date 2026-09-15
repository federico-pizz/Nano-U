"""Benchmark host-side inference latency for a model.

This is the number that contextualises the on-device figure: the same model,
measured on the host, so the MCU latency has something to be compared against.

Usage:
    python scripts/bench.py nanou
    python scripts/bench.py bu_net --preset botanicgarden
"""

import argparse

from nanou.bench import run_benchmarks
from nanou.config import load_config, preset_path
from nanou.preflight import PreflightError, require_keras_model


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nano-U or BU-Net model inference speed")
    parser.add_argument("model", choices=["bu_net", "nanou"], help="Model to benchmark")
    parser.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    parser.add_argument(
        "--config", default=None, help="explicit config file, used instead of --preset"
    )
    args = parser.parse_args()
    config_path = args.config or str(preset_path(args.preset))

    print(f"{'=' * 55}")
    print(f"BENCHMARKING MODEL: {args.model}")
    print(f"{'=' * 55}")

    try:
        require_keras_model(
            load_config(config_path)["data"]["paths"]["models_dir"], args.model, preset=args.preset
        )
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    inf = run_benchmarks(args.model, config_path=config_path)["inference"]
    print("\nFinal Metrics:")
    print(f"  Avg Latency:    {inf.get('avg_latency_ms', '?'):.2f} ms")
    print(f"  P99 Latency:    {inf.get('p99_latency_ms', '?'):.2f} ms")
    print(f"  Throughput:     {inf.get('throughput_fps', '?'):.1f} FPS")
    print(f"  Total Time:     {inf.get('total_time_sec', '?'):.2f} sec")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
