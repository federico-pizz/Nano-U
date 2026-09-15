"""Stack-usage and power profiling over the device serial log."""

import matplotlib.pyplot as plt

from nanou.device.session import DeviceSession

TIMEOUT_SECONDS = 300


def _read_power(preset_voltage, preset_current):
    """Supply voltage and current: the presets when given, otherwise asked for at the terminal."""
    print("\n" + "=" * 60)
    print("DEVICE IS NOW UNDER 100% CONTINUOUS NEURAL NETWORK LOAD")
    print("=" * 60)
    print("The ESP32 is running a silent infinite loop.")

    current_ma = None
    if preset_voltage is not None:
        voltage_v = preset_voltage
        print(f"Using preset voltage: {voltage_v} V")
    else:
        print("Look at your multimeter and wait for the reading to stabilize.\n")
        raw = input("Enter the measured voltage in V (or press Enter to use 5.0V): ").strip()
        voltage_v = 5.0
        if raw:
            try:
                voltage_v = float(raw)
            except ValueError:
                print("Invalid input, defaulting to 5.0V.")

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
    return voltage_v, current_ma


def run_analysis_on_device(cmd, log_file, cwd=None, preset_voltage=None, preset_current=None):
    """Run the analysis binary until it starts its power loop, teeing its output to a log file.

    Args:
        cmd: Command to run, as an argv list.
        log_file: Path the combined output is written to; `parse_log_output` reads it.
        cwd: Working directory for the command.
        preset_voltage: Supply voltage in volts, when measured externally.
        preset_current: Supply current in milliamps, when measured externally.

    Returns:
        Tuple of (success, current_ma, voltage_v). False means the board never
        reached POWER_MEASUREMENT_START; the caller decides whether to retry.
    """
    print(f"Executing: {' '.join(cmd)}")
    with DeviceSession(cmd, cwd=cwd, log_path=log_file, timeout=TIMEOUT_SECONDS) as session:
        for line in session.lines():
            if "POWER_MEASUREMENT_START" in line:
                voltage_v, current_ma = _read_power(preset_voltage, preset_current)
                return True, current_ma, voltage_v
        if session.timed_out:
            print(
                f"\nTimeout: device did not reach POWER_MEASUREMENT_START within {TIMEOUT_SECONDS}s."
            )
        else:
            print("\nProcess exited before POWER_MEASUREMENT_START.")
    return False, None, None


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
