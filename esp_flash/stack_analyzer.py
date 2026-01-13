
#!/usr/bin/env python3
"""
Parse ESP32 Stack Painting output and visualize usage.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_output(filename):
    """
    Parse UART log file to extract stack usage data.
    Returns a list of peak usages and the total stack size.
    """
    peaks = []
    total_stack = 0
    
    with open(filename, 'r') as f:
        for line in f:
            if "STACK_PEAK:" in line:
                try:
                    val = int(line.split("STACK_PEAK:")[1].strip())
                    peaks.append(val)
                except ValueError:
                    pass
            if "STACK_TOTAL:" in line:
                try:
                    # Update total_stack (assuming it's constant)
                    total_stack = int(line.split("STACK_TOTAL:")[1].strip())
                except ValueError:
                    pass
                    
    return peaks, total_stack

def plot_stack_usage(peaks, total_stack, output_file='stack_usage.png'):
    """
    Create visualization of stack usage.
    """
    if not peaks:
        print("No valid stack usage data found!")
        return
        
    peak_usage = max(peaks)
    
    # If total_stack wasn't found, assume a default or calculate from peak
    if total_stack == 0:
        total_stack = max(peak_usage * 1.2, 320 * 1024) # Default to 320KB if unknown
    
    free_stack = total_stack - peak_usage
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Usage over Iterations
    iterations = range(1, len(peaks) + 1)
    ax1.plot(iterations, [p/1024 for p in peaks], 'o-', color='#2c3e50', linewidth=2)
    ax1.set_title('Stack Usage over Iterations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Stack Usage (KB)')
    ax1.set_ylim(0, total_stack/1024 * 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line for Total Stack
    ax1.axhline(y=total_stack/1024, color='r', linestyle='--', label='Total Stack Limit')
    ax1.legend()
    
    # Plot 2: Peak Usage Bar Chart
    # Data
    labels = ['Used Stack', 'Free Stack']
    
    # Bar Chart
    ax2.bar(['Stack Memory'], [total_stack/1024], color='#ecf0f1', edgecolor='black', label='Free Space')
    ax2.bar(['Stack Memory'], [peak_usage/1024], color='#e74c3c', label='Peak Used')
    
    # Add text labels
    ax2.text(0, peak_usage/1024 / 2, f'{peak_usage/1024:.1f} KB\n({peak_usage/total_stack*100:.1f}%)', 
            ha='center', va='center', color='white', fontweight='bold')
    ax2.text(0, (peak_usage + free_stack/2)/1024, f'Free: {free_stack/1024:.1f} KB', 
            ha='center', va='center', color='black')
            
    ax2.set_ylabel('Memory (KB)', fontsize=12)
    ax2.set_title('Peak Stack Memory Usage', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Add Model Info Text to the figure
    model_info = (
        f"Analysis Summary:\n"
        f"  Iterations: {len(peaks)}\n"
        f"  Peak Usage: {peak_usage} bytes\n"
        f"  Total Size: {total_stack} bytes\n"
        f"  Headroom:   {free_stack} bytes"
    )
    plt.figtext(0.5, 0.02, model_info, fontsize=11, ha='center',
                bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # Make room for text
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")

def main():
    log_file = 'stack_log.txt'
    
    if not Path(log_file).exists():
        print(f"Error: {log_file} not found!")
        return
    
    print(f"Parsing {log_file}...")
    peaks, total = parse_log_output(log_file)
    
    if not peaks:
        print("No stack peaks found in log.")
        return

    peak = max(peaks)
    print(f"Peak Stack Usage: {peak} bytes ({peak/1024:.2f} KB)")
    print(f"Total Stack Size: {total} bytes ({total/1024:.2f} KB)")
    
    plot_stack_usage(peaks, total)

if __name__ == '__main__':
    main()
