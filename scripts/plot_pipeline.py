import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_box(ax, center, size, title, content, edgecolor='black', facecolor='white', title_color='black', content_color='#333333'):
    # Main container box
    box = patches.FancyBboxPatch((center[0] - size[0]/2, center[1] - size[1]/2), size[0], size[1],
                                 boxstyle="round,pad=0.1",
                                 edgecolor=edgecolor, facecolor=facecolor, lw=1.5, zorder=2)
    ax.add_patch(box)
    
    # Title (Bold, larger, top-aligned inside box)
    ax.text(center[0], center[1] + size[1]/2 - 0.25, title, ha='center', va='top', 
            color=title_color, fontsize=11, weight='bold', zorder=3)
            
    # Content (Regular, smaller, body-aligned)
    lines = content.split('\n')
    y_start = center[1] + size[1]/2 - 0.75
    
    for i, line in enumerate(lines):
        ax.text(center[0], y_start - i*0.28, line, ha='center', va='center', 
                color=content_color, fontsize=9.5, zorder=3)

def draw_arrow(ax, start, end, text=None):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, shrinkA=0, shrinkB=0),
                zorder=1)
                
    if text:
        mid_x = (start[0] + end[0]) / 2 + 0.1
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, text, ha='left', va='center', fontsize=9, color='#333333', 
                style='italic', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.5))

def main():
    # Adjusted for a half-page column width (approx 3.5 inches wide, taller aspect ratio)
    fig, ax = plt.subplots(figsize=(3.8, 7.0))
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Define box centers, sizes, and detailed texts for VERTICAL layout
    boxes = [
        {
            "center": (1.9, 7.9), 
            "size": (3.4, 2.0), 
            "title": "1. Constraint-Driven Training", 
            "content": "Architecture: Nano-U\nQuantization-Aware Distillation (QAD)\nParams: ~3.3K (Depthwise Convs)\nDataset: Botanic Garden", 
            "fc": "#f0f8ff", 
            "ec": "#0055a4",
            "tc": "#003366"
        },
        {
            "center": (1.9, 4.7), 
            "size": (3.4, 2.0), 
            "title": "2. PTQ & Code Generation", 
            "content": "TFLite INT8 Quantization (34 KB)\nRust `build.rs` Extraction\nCompile-Time Graph Evaluation\nZero-Allocation Statically Built", 
            "fc": "#fff5eb", 
            "ec": "#cc5500",
            "tc": "#803300"
        },
        {
            "center": (1.9, 1.5), 
            "size": (3.4, 2.0), 
            "title": "3. Bare-Metal Execution", 
            "content": "Hardware: ESP32-S3-CAM\nEngine: microflow-rs (no_std)\nIRAM Hot Loop Pinning\nPeak DRAM: 257 KB", 
            "fc": "#f0fff0", 
            "ec": "#008000",
            "tc": "#004d00"
        },
    ]

    # Draw arrows with labels (vertical, flowing down)
    
    # Dataset to training
    draw_arrow(ax, (1.9, 9.3), (1.9, 8.9), "Domain-Specific Data")
    
    # Training to Quantization/Compilation
    draw_arrow(ax, (1.9, 6.9), (1.9, 5.7), "Float32 Weights (.keras)")
    
    # Quantization to Deployment
    draw_arrow(ax, (1.9, 3.7), (1.9, 2.5), "Statically Linked Firmware")
    
    # Output from edge
    draw_arrow(ax, (1.9, 0.5), (1.9, 0.1), "Binary Traverse Mask")

    # Draw boxes
    for b in boxes:
        draw_box(ax, b["center"], b["size"], b["title"], b["content"], 
                 edgecolor=b["ec"], facecolor=b["fc"], title_color=b["tc"])

    plt.tight_layout()
    
    # Save the figure
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_summary_paper_column.png"
    out_pdf = out_dir / "pipeline_summary_paper_column.pdf"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Paper-ready column figures saved to:\n  - {out_path.resolve()}\n  - {out_pdf.resolve()}")

if __name__ == "__main__":
    main()
