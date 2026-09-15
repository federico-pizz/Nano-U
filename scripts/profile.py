"""Stack and power profiling of the firmware on an attached device."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from nanou.device.profile import plot_stack_usage, run_analysis_on_device
from nanou.device.protocol import parse_log_output


def main():
    parser = argparse.ArgumentParser(description="Profile Nano-U on ESP32 hardware")
    parser.add_argument(
        "--voltage", type=float, default=None, help="Supply voltage in V (skips interactive prompt)"
    )
    parser.add_argument(
        "--current", type=float, default=None, help="Current draw in mA (skips interactive prompt)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    out_dir = repo_root / "results" / "hardware" / "nanou"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "stack_log.txt"

    print("[1/3] Compiling analysis binary...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "analysis"],
        cwd=repo_root / "firmware",
        check=True,
    )

    print("\n[2/3] Flashing and running analysis...")
    cmd = ["cargo", "run", "--release", "--bin", "analysis"]

    success = False
    current_ma = None
    voltage_v = None
    for attempt in range(3):
        success, current_ma, voltage_v = run_analysis_on_device(
            cmd,
            str(log_file),
            cwd=repo_root / "firmware",
            preset_voltage=args.voltage,
            preset_current=args.current,
        )
        if success:
            break
        print(f"Attempt {attempt + 1} failed. Retrying in 2s...")
        time.sleep(2)

    if not success:
        print("Failed to complete analysis on device.")
        sys.exit(1)

    print("\n[3/3] Parsing output and rendering plots...")
    peaks, total, avg_time_ms, all_times = parse_log_output(str(log_file))

    if not peaks:
        print("No STACK_PEAK entries found in log. Check firmware output.")
        sys.exit(1)

    peak = max(peaks)
    print(f"Stack peak:          {peak} bytes ({peak / 1024:.2f} KB)")
    if total:
        print(f"Stack total:         {total} bytes ({total / 1024:.2f} KB)")
        print(f"Headroom:            {total - peak} bytes ({(total - peak) / 1024:.2f} KB)")
    if all_times:
        print(f"Inference samples:   {len(all_times)}")
        print(f"Avg inference time:  {avg_time_ms:.2f} ms")
        print(f"Min / Max:           {min(all_times):.2f} ms / {max(all_times):.2f} ms")

    plot_stack_usage(peaks, total, output_file=str(out_dir / "stack_usage.png"))

    metrics = {
        "peak_stack_bytes": peak,
        "peak_stack_kb": round(peak / 1024, 2),
        "total_stack_bytes": total,
        "headroom_bytes": total - peak if total else None,
        "inference_samples": len(all_times),
        "avg_inference_time_ms": round(avg_time_ms, 3),
        "min_inference_time_ms": round(min(all_times), 3) if all_times else None,
        "max_inference_time_ms": round(max(all_times), 3) if all_times else None,
    }

    if current_ma is not None and voltage_v is not None:
        power_watts = (current_ma / 1000) * voltage_v
        power_mw = power_watts * 1000
        energy_joules = power_watts * (avg_time_ms / 1000)
        energy_mj = energy_joules * 1000

        print(f"\n--- Energy Estimates (at {voltage_v}V) ---")
        print(f"Active power draw:        {power_mw:.2f} mW")
        print(f"Energy per inference:     {energy_mj:.3f} mJ  ({energy_joules:.6f} J)")
        print("Note: these are board-level figures (USB bridge + LDO losses included).")

        metrics["active_current_ma"] = current_ma
        metrics["voltage_v"] = voltage_v
        metrics["active_power_mw"] = round(power_mw, 3)
        metrics["energy_per_inference_mj"] = round(energy_mj, 4)
        metrics["energy_per_inference_j"] = round(energy_joules, 7)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    # Ctrl-C is how a profiling run is normally stopped at the multimeter
    # prompt, so it gets a clean exit rather than a traceback.
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
