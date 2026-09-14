"""Stack-usage and power profiling over the device serial log."""

import os
import pty
import select
import subprocess
import sys
import time

import matplotlib.pyplot as plt

from nanou.device.session import drain_pty

TIMEOUT_SECONDS = 300


def run_analysis_on_device(cmd, log_file, cwd=None, preset_voltage=None, preset_current=None):
    """Run a flash-and-monitor command under a pty and tee the board's output to a log file.

    A pty is needed because the cargo runner only streams the monitor output
    when it believes it is attached to a terminal.

    Args:
        cmd: Command to run, as an argv list.
        log_file: Path the combined output is written to.
        cwd: Working directory for the command.
        preset_voltage: Supply voltage in volts, when measured externally.
        preset_current: Supply current in milliamps, when measured externally.

    Returns:
        Tuple of (success, current_ma, voltage_v).
    """
    print(f"Executing: {' '.join(cmd)}")
    master, slave = pty.openpty()
    p = subprocess.Popen(
        cmd, stdin=slave, stdout=slave, stderr=subprocess.STDOUT, close_fds=True, cwd=cwd
    )
    os.close(slave)

    current_ma = None
    voltage_v = None
    success = False

    with open(log_file, "wb") as f_log:
        buffer = b""
        start_time = time.time()

        try:
            while True:
                if time.time() - start_time > TIMEOUT_SECONDS:
                    print(
                        f"\nTimeout: device did not reach POWER_MEASUREMENT_START within {TIMEOUT_SECONDS}s."
                    )
                    break

                if p.poll() is not None:
                    drain_pty(master, f_log)
                    print("\nProcess exited before POWER_MEASUREMENT_START.")
                    break

                r, _, _ = select.select([master], [], [], 0.1)
                if master not in r:
                    continue

                try:
                    data = os.read(master, 1024)
                except OSError:
                    break
                if not data:
                    break

                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except Exception:
                    pass

                f_log.write(data)
                f_log.flush()
                buffer += data

                if b"POWER_MEASUREMENT_START" in buffer:
                    print("\n" + "=" * 60)
                    print("DEVICE IS NOW UNDER 100% CONTINUOUS NEURAL NETWORK LOAD")
                    print("=" * 60)
                    print("The ESP32 is running a silent infinite loop.")

                    if preset_voltage is not None:
                        voltage_v = preset_voltage
                        print(f"Using preset voltage: {voltage_v} V")
                    else:
                        print("Look at your multimeter and wait for the reading to stabilize.\n")
                        raw = input(
                            "Enter the measured voltage in V (or press Enter to use 5.0V): "
                        ).strip()
                        if raw:
                            try:
                                voltage_v = float(raw)
                            except ValueError:
                                print("Invalid input, defaulting to 5.0V.")
                                voltage_v = 5.0
                        else:
                            voltage_v = 5.0

                    if preset_current is not None:
                        current_ma = preset_current
                        print(f"Using preset current: {current_ma} mA")
                    elif preset_voltage is None:
                        raw = input(
                            "Enter the stable current draw in mA (or press Enter to skip energy calc): "
                        ).strip()
                        if raw:
                            try:
                                current_ma = float(raw)
                            except ValueError:
                                print("Invalid input. Skipping energy calculation.")
                    else:
                        print("No current provided - skipping energy calculation.")

                    success = True
                    break

        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(130)
        finally:
            if p.poll() is None:
                p.terminate()
                time.sleep(0.5)
                if p.poll() is None:
                    p.kill()
            os.close(master)

    return success, current_ma, voltage_v


def plot_stack_usage(peaks, total_stack, output_file="stack_usage.png"):
    """Plot per-task peak stack usage against the total stack, and save it to `output_file`."""
    if not peaks:
        return

    peak_usage = max(peaks)

    if total_stack == 0:
        print(
            "WARNING: STACK_TOTAL not found in log. The stack limit line is an estimate (peak * 1.5)."
        )
        print("         Emit 'STACK_TOTAL: <bytes>' from firmware for accurate headroom.")
        total_stack = int(peak_usage * 1.5)

    if peak_usage > total_stack:
        print(
            f"WARNING: peak usage ({peak_usage}) exceeds reported total stack ({total_stack}). "
            "Check your STACK_TOTAL firmware output."
        )

    free_stack = total_stack - peak_usage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    iterations = range(1, len(peaks) + 1)
    ax1.plot(iterations, [p / 1024 for p in peaks], "o-", color="#2c3e50", linewidth=2)
    ax1.set_title("Stack Usage over Iterations", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Stack Usage (KB)")
    ax1.set_ylim(0, total_stack / 1024 * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=total_stack / 1024,
        color="r",
        linestyle="--",
        label="Total Stack Limit" + (" (estimated)" if free_stack < 0 else ""),
    )
    ax1.legend()

    ax2.bar(
        ["Stack Memory"],
        [total_stack / 1024],
        color="#ecf0f1",
        edgecolor="black",
        label="Free Space",
    )
    ax2.bar(["Stack Memory"], [peak_usage / 1024], color="#e74c3c", label="Peak Used")
    ax2.text(
        0,
        peak_usage / 1024 / 2,
        f"{peak_usage / 1024:.1f} KB\n({peak_usage / total_stack * 100:.1f}%)",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax2.text(
        0,
        (peak_usage + free_stack / 2) / 1024,
        f"Free: {free_stack / 1024:.1f} KB",
        ha="center",
        va="center",
        color="black",
    )

    ax2.set_ylabel("Memory (KB)", fontsize=12)
    ax2.set_title("Peak Stack Memory Usage", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
