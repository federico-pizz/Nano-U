#!/usr/bin/env python3
import subprocess
import os
import sys
import pty
import select
import time
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path

def run_analysis_on_device(cmd, log_file, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, stdout=slave, stderr=subprocess.STDOUT, close_fds=True, cwd=cwd)
    os.close(slave)
    
    current_ma = None
    success = False

    with open(log_file, "wb") as f_log:
        buffer = b""
        start_time = time.time()
        
        try:
            while True:
                if p.poll() is not None:
                    break
                r, _, _ = select.select([master], [], [], 0.1)
                if master in r:
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
                    
                    # Wait until the device enters the silent infinite loop
                    if b"POWER_MEASUREMENT_START" in buffer:
                        print("\n" + "="*60)
                        print("⚡ DEVICE IS NOW UNDER 100% CONTINUOUS NEURAL NETWORK LOAD ⚡")
                        print("="*60)
                        print("The ESP32 is running a silent infinite loop.")
                        print("Please look at your multimeter. Wait for the reading to stabilize.")
                        
                        user_input = input("\nEnter the stable current draw in mA (or press Enter to skip): ").strip()
                        if user_input:
                            try:
                                current_ma = float(user_input)
                            except ValueError:
                                print("Invalid input. Skipping energy calculation.")
                        
                        success = True
                        break # Now we can safely kill espflash
            
            if time.time() - start_time > 300 and not success:
                print("\nTimeout reached!")
                
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
            
    return success, current_ma

def parse_log_output(filename):
    peaks = []
    inf_times = []
    total_stack = 0
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    
    with open(filename, 'r') as f:
        for line in f:
            line = ansi_escape.sub('', line).strip()
            
            if "STACK_PEAK:" in line:
                try:
                    peaks.append(int(line.split("STACK_PEAK:")[1].strip()))
                except ValueError: pass
                
            if "STACK_TOTAL:" in line:
                try:
                    total_stack = int(line.split("STACK_TOTAL:")[1].strip())
                except ValueError: pass
                
            if "Inference done in" in line:
                match = re.search(r'Inference done in (\d+) ms', line)
                if match:
                    inf_times.append(int(match.group(1)))
                    
    avg_time = sum(inf_times) / len(inf_times) if inf_times else 0
    return peaks, total_stack, avg_time

def plot_stack_usage(peaks, total_stack, output_file='stack_usage.png'):
    if not peaks:
        return
        
    peak_usage = max(peaks)
    if total_stack == 0:
        total_stack = max(peak_usage * 1.2, 320 * 1024) 
    
    free_stack = total_stack - peak_usage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    iterations = range(1, len(peaks) + 1)
    ax1.plot(iterations, [p/1024 for p in peaks], 'o-', color='#2c3e50', linewidth=2)
    ax1.set_title('Stack Usage over Iterations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Stack Usage (KB)')
    ax1.set_ylim(0, total_stack/1024 * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=total_stack/1024, color='r', linestyle='--', label='Total Stack Limit')
    ax1.legend()
    
    ax2.bar(['Stack Memory'], [total_stack/1024], color='#ecf0f1', edgecolor='black', label='Free Space')
    ax2.bar(['Stack Memory'], [peak_usage/1024], color='#e74c3c', label='Peak Used')
    ax2.text(0, peak_usage/1024 / 2, f'{peak_usage/1024:.1f} KB\n({peak_usage/total_stack*100:.1f}%)', 
            ha='center', va='center', color='white', fontweight='bold')
    ax2.text(0, (peak_usage + free_stack/2)/1024, f'Free: {free_stack/1024:.1f} KB', 
            ha='center', va='center', color='black')
            
    ax2.set_ylabel('Memory (KB)', fontsize=12)
    ax2.set_title('Peak Stack Memory Usage', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

def main():
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    out_dir = repo_root / 'results' / 'nano_u_rust'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'stack_log.txt'
    
    print("[1/3] Compiling analysis binary...")
    subprocess.run(["cargo", "build", "--release", "--bin", "analysis"], cwd=repo_root / "esp_flash", check=True)
    
    print("\n[2/3] Flashing and running analysis...")
    cmd = ["espflash", "flash", "--monitor", "target/xtensa-esp32s3-none-elf/release/analysis"]
    
    success = False
    current_ma = None
    for attempt in range(3):
        success, current_ma = run_analysis_on_device(cmd, str(log_file), cwd=repo_root / 'esp_flash')
        if success:
            break
        print("Retrying...")
        time.sleep(2)
        
    if not success:
        print("Failed to complete analysis on device.")
        sys.exit(1)
        
    print("\n[3/3] Parsing output and rendering plots...")
    peaks, total, avg_time_ms = parse_log_output(str(log_file))
    
    if peaks:
        peak = max(peaks)
        print(f"Peak Stack Usage: {peak} bytes ({peak/1024:.2f} KB)")
        print(f"Average Inference Time: {avg_time_ms:.1f} ms")
        plot_stack_usage(peaks, total, output_file=str(out_dir / 'stack_usage.png'))
        
        metrics = {
            'peak_stack_bytes': peak,
            'peak_stack_kb': round(peak / 1024, 2),
            'total_stack_bytes': total,
            'headroom_bytes': total - peak,
            'avg_inference_time_ms': round(avg_time_ms, 2)
        }
        
        # --- Energy Calculation ---
        if current_ma is not None:
            # Note: We assume 5.0V since you are measuring via the USB cable from your PC.
            # If measuring directly via the 3.3V pin, change this to 3.3
            VOLTAGE = 5.0 
            
            power_watts = (current_ma / 1000) * VOLTAGE
            power_mw = power_watts * 1000
            
            # Energy (Joules) = Power (W) * Time (s)
            energy_joules = power_watts * (avg_time_ms / 1000)
            energy_mj = energy_joules * 1000
            
            print(f"\n--- Energy Estimates (at {VOLTAGE}V) ---")
            print(f"Active Power Draw: {power_mw:.2f} mW")
            print(f"Energy per Inference: {energy_mj:.2f} mJ ({energy_joules:.5f} Joules)")
            
            metrics['active_current_ma'] = current_ma
            metrics['voltage'] = VOLTAGE
            metrics['active_power_mw'] = round(power_mw, 2)
            metrics['energy_per_inference_mj'] = round(energy_mj, 3)

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
if __name__ == '__main__':
    main()