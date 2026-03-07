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
    
    with open(log_file, "wb") as f_log:
        buffer = b""
        start_time = time.time()
        success = False
        
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
                    
                    sys.stdout.write(data.decode('utf-8', errors='replace'))
                    sys.stdout.flush()
                    f_log.write(data)
                    f_log.flush()
                    
                    buffer += data
                    if b"ANALYSIS_DONE" in buffer:
                        success = True
                        break
            
            if time.time() - start_time > 300:
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
    return success

def parse_log_output(filename):
    peaks = []
    total_stack = 0
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    with open(filename, 'r') as f:
        for line in f:
            line = ansi_escape.sub('', line).strip()
            if "STACK_PEAK:" in line:
                try:
                    peaks.append(int(line.split("STACK_PEAK:")[1].strip()))
                except ValueError:
                    pass
            if "STACK_TOTAL:" in line:
                try:
                    total_stack = int(line.split("STACK_TOTAL:")[1].strip())
                except ValueError:
                    pass
    return peaks, total_stack

def plot_stack_usage(peaks, total_stack, output_file='stack_usage.png'):
    if not peaks:
        print("No valid data found!")
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
    
    model_info = (
        f"Analysis Summary:\n  Iterations: {len(peaks)}\n"
        f"  Peak Usage: {peak_usage} bytes\n  Total Size: {total_stack} bytes\n"
        f"  Headroom:   {free_stack} bytes"
    )
    plt.figtext(0.5, 0.02, model_info, fontsize=11, ha='center',
                bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
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
    for attempt in range(3):
        if run_analysis_on_device(cmd, str(log_file), cwd=repo_root / 'esp_flash'):
            success = True
            break
        print("Retrying...")
        time.sleep(2)
        
    if not success:
        print("Failed to complete analysis on device.")
        sys.exit(1)
        
    print("\n[3/3] Parsing output and rendering plots...")
    peaks, total = parse_log_output(str(log_file))
    if peaks:
        peak = max(peaks)
        print(f"Peak Stack Usage: {peak} bytes ({peak/1024:.2f} KB)")
        
        plot_stack_usage(peaks, total, output_file=str(out_dir / 'stack_usage.png'))
        
        metrics = {
            'peak_stack_bytes': peak,
            'peak_stack_kb': round(peak / 1024, 2),
            'total_stack_bytes': total,
            'total_stack_kb': round(total / 1024, 2),
            'headroom_bytes': total - peak,
            'iterations': len(peaks),
            'model': 'nano_u'
        }
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
if __name__ == '__main__':
    main()
