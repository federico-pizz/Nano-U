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
                        break 
            
            if time.time() - start_time > 300 and not success:
                print("\nTimeout reached!")
                
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            # Do NOT sys.exit here, instead set flag so we can save partial data
            success = True
            current_ma = None
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

def main():
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    
    # Check if we have the assets ready inside esp_flash/models
    if not (repo_root / 'esp_flash' / 'models' / 'person_detect.tflite').exists():
        print("person_detect.tflite not found in esp_flash/models/ directory!")
        sys.exit(1)
        
    if not (repo_root / 'esp_flash' / 'models' / 'test_img.bmp').exists():
        print("test_img.bmp not found in esp_flash/models/ directory!")
        sys.exit(1)
        
    out_dir = repo_root / 'results' / 'person_detect'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'stack_log.txt'
    
    print("[1/3] Compiling analysis_person_detect binary...")
    subprocess.run(["cargo", "build", "--release", "--bin", "analysis_person_detect"], cwd=repo_root / "esp_flash", check=True)
    
    print("\n[2/3] Flashing and running analysis...")
    cmd = ["espflash", "flash", "--monitor", "target/xtensa-esp32s3-none-elf/release/analysis_person_detect"]
    
    success = False
    current_ma = None
    try:
        for attempt in range(3):
            success, current_ma = run_analysis_on_device(cmd, str(log_file), cwd=repo_root / 'esp_flash')
            if success:
                break
            print("Retrying...")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nSkipping further retries due to interrupt.")
        
    if not success and not log_file.exists():
        print("Failed to complete analysis on device.")
        sys.exit(1)
        
    print("\n[3/3] Parsing output and logging metrics...")
    peaks, total, avg_time_ms = parse_log_output(str(log_file))
    
    if peaks:
        peak = max(peaks)
        print(f"\n--- {('PERSON DETECT PERFORMANCE')} ---")
        print(f"Peak Stack Usage: {peak} bytes ({peak/1024:.2f} KB)")
        print(f"Average Inference Time: {avg_time_ms:.1f} ms")
        
        metrics = {
            'peak_stack_bytes': peak,
            'peak_stack_kb': round(peak / 1024, 2),
            'total_stack_bytes': total,
            'headroom_bytes': total - peak,
            'avg_inference_time_ms': round(avg_time_ms, 2)
        }
        
        if current_ma is not None:
            VOLTAGE = 5.0 
            power_watts = (current_ma / 1000) * VOLTAGE
            power_mw = power_watts * 1000
            energy_joules = power_watts * (avg_time_ms / 1000)
            energy_mj = energy_joules * 1000
            
            print(f"Active Power Draw: {power_mw:.2f} mW")
            print(f"Energy per Inference: {energy_mj:.2f} mJ")
            
            metrics['active_current_ma'] = current_ma
            metrics['voltage'] = VOLTAGE
            metrics['active_power_mw'] = round(power_mw, 2)
            metrics['energy_per_inference_mj'] = round(energy_mj, 3)

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nResults saved to {out_dir}")

if __name__ == '__main__':
    main()
