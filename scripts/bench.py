"""Benchmark host-side inference latency for a model.

This is the number that contextualises the on-device figure: the same model,
measured on the host, so the MCU latency has something to be compared against.

Usage:
    python scripts/bench.py nano_u
    python scripts/bench.py bu_net --config config/TinyAgri_config.yaml
"""

import argparse
import sys

from src.benchmarks import run_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Nano-U or BU-Net model inference speed')
    parser.add_argument('model', choices=['bu_net', 'nano_u'], help='Model to benchmark')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f"BENCHMARKING MODEL: {args.model}")
    print(f"{'='*55}")

    results = run_benchmarks(args.model, config_path=args.config)
    if "error" in results:
        print(f"\nBenchmark Failed: {results['error']}")
        print(results.get("traceback", ""))
        sys.exit(1)

    inf = results.get("inference", {})
    print("\nFinal Metrics:")
    print(f"  Avg Latency:    {inf.get('avg_latency_ms', '?'):.2f} ms")
    print(f"  P99 Latency:    {inf.get('p99_latency_ms', '?'):.2f} ms")
    print(f"  Throughput:     {inf.get('throughput_fps', '?'):.1f} FPS")
    print(f"  Total Time:     {inf.get('total_time_sec', '?'):.2f} sec")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
