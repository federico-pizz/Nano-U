#!/usr/bin/env python3
"""Scrape all result artefacts from results/ and write a single Markdown report."""

import csv
import json
import re
import sys
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)

def pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def ms(v: float) -> str:
    return f"{v:.0f} ms"

def kb(v: float) -> str:
    return f"{v / 1024:.1f} KB"

def _row(cells: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [_row(headers, widths), sep]
    for row in rows:
        lines.append(_row(row, widths))
    return "\n".join(lines)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_training_results(results_path: Path) -> dict | None:
    if not results_path.exists():
        return None
    data = load_json(results_path)
    if data.get("status") != "success":
        return None
    m = data.get("final_metrics", {})
    return {
        "train_miou":    m.get("binary_iou"),
        "train_loss":    m.get("loss"),
        "val_miou":      m.get("val_binary_iou"),
        "val_loss":      m.get("val_loss"),
        "model_path":    data.get("model_path", ""),
    }


def parse_eval_results(results_path: Path) -> dict | None:
    """Load float32 test-set metrics written by evaluate.py.

    Looks for eval_results.json (saved by evaluate.py alongside the prediction
    plot) first. Falls back to results_path only when that file already
    contains a top-level 'miou' key (i.e. it was written by evaluate.py
    directly).  The training results.json (which holds val-set metrics under
    'final_metrics') is intentionally *not* used here, because comparing
    val-set float32 mIoU against test-set INT8 mIoU gives a misleading
    quantization-gap figure.
    """
    # Preferred: dedicated eval_results.json produced by evaluate.py
    eval_path = results_path.parent / "eval_results.json"
    if eval_path.exists():
        data = load_json(eval_path)
        if "miou" in data:
            return data

    # Legacy: results.json written directly by evaluate.py (has top-level 'miou')
    if results_path.exists():
        data = load_json(results_path)
        if "miou" in data:
            return data

    return None


def parse_hw_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def parse_hardware_profile(metrics_path: Path, stack_log_path: Path) -> dict:
    """Merge person_detect/metrics.json with raw values from stack_log.txt."""
    result = {}

    if metrics_path.exists():
        result.update(load_json(metrics_path))

    if stack_log_path.exists():
        text = stack_log_path.read_text(errors="replace")
        # Strip ANSI codes
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)

        peaks, totals, times = [], [], []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("STACK_PEAK:"):
                try:
                    peaks.append(int(line.split(":")[1]))
                except ValueError:
                    pass
            elif line.startswith("STACK_TOTAL:"):
                try:
                    totals.append(int(line.split(":")[1]))
                except ValueError:
                    pass
            elif m := re.match(r"Inference done in (\d+) ms", line):
                times.append(int(m.group(1)))

        if peaks:
            result.setdefault("peak_stack_bytes", peaks[0])
        if totals:
            result.setdefault("total_stack_bytes", totals[0])
        if times:
            result.setdefault("avg_inference_time_ms", sum(times) / len(times))

    return result


def parse_nas_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ── Section builders ──────────────────────────────────────────────────────────

DATASETS = ["botanic_garden", "tinyagri"]
MODELS   = ["bu_net", "nano_u"]

def section_training(results_root: Path) -> str:
    lines = ["## Training — Final Metrics\n"]

    for model in MODELS:
        lines.append(f"### {model}\n")
        rows = []
        for dataset in DATASETS:
            d = parse_training_results(results_root / dataset / model / "results.json")
            if d is None:
                continue
            rows.append([
                dataset.replace('_', ' ').title(),
                pct(d["train_miou"])  if d.get("train_miou")  is not None else "—",
                f"{d['train_loss']:.4f}" if d.get("train_loss") is not None else "—",
                pct(d["val_miou"])    if d.get("val_miou")    is not None else "—",
                f"{d['val_loss']:.4f}" if d.get("val_loss")   is not None else "—",
            ])
        if rows:
            lines.append(markdown_table(
                ["Dataset", "Train mIoU", "Train Loss", "Val mIoU", "Val Loss"],
                rows,
            ))
        lines.append("")

    return "\n".join(lines)


def section_eval(results_root: Path) -> str:
    lines = ["## Test-Set Evaluation (Float32)\n"]

    all_rows = []
    for model in MODELS:
        for dataset in DATASETS:
            rpath = results_root / dataset / model / "results.json"
            d = parse_eval_results(rpath)
            if d is None:
                continue
            all_rows.append([
                model,
                dataset.replace("_", " ").title(),
                pct(d.get("miou", 0)),
                pct(d.get("f1", 0)),
                pct(d.get("precision", 0)),
                pct(d.get("recall", 0)),
            ])

    if all_rows:
        lines.append(markdown_table(
            ["Model", "Dataset", "mIoU", "F1", "Precision", "Recall"],
            all_rows,
        ))
    else:
        lines.append("_No evaluation results found._")
    lines.append("")
    return "\n".join(lines)


def section_hw_eval(results_root: Path) -> str:
    lines = ["## On-Device Evaluation (INT8, ESP32-S3)\n"]

    all_rows = []
    for dataset in DATASETS:
        hw = parse_hw_metrics(results_root / dataset / "nano_u" / "hw_metrics.json")
        if hw is None:
            continue
        all_rows.append([
            dataset.replace("_", " ").title(),
            pct(hw.get("miou", 0)),
            pct(hw.get("f1", 0)),
            pct(hw.get("precision", 0)),
            pct(hw.get("recall", 0)),
        ])

    if all_rows:
        lines.append(markdown_table(
            ["Dataset", "mIoU", "F1", "Precision", "Recall"],
            all_rows,
        ))
    else:
        lines.append("_No on-device evaluation results found._")
    lines.append("")
    return "\n".join(lines)


def section_quantization_gap(results_root: Path) -> str:
    """Side-by-side float32 vs INT8 mIoU to show the quantization cost."""
    lines = ["## Quantization Gap (Float32 → INT8)\n"]

    rows = []
    for dataset in DATASETS:
        f32 = parse_eval_results(results_root / dataset / "nano_u" / "results.json")
        i8  = parse_hw_metrics(results_root / dataset / "nano_u" / "hw_metrics.json")
        if f32 is None or i8 is None:
            continue
        f32_miou = f32.get("miou", 0)
        i8_miou  = i8.get("miou", 0)
        delta    = i8_miou - f32_miou
        sign     = "+" if delta >= 0 else ""
        rows.append([
            dataset.replace("_", " ").title(),
            pct(f32_miou),
            pct(i8_miou),
            f"{sign}{delta * 100:.1f} pp",
        ])

    if rows:
        lines.append(markdown_table(
            ["Dataset", "Float32 mIoU", "INT8 mIoU", "Δ"],
            rows,
        ))
    else:
        lines.append("_Not enough data to compute quantization gap._")
    lines.append("")
    return "\n".join(lines)


def _hw_table(hw: dict) -> str:
    """Render a hardware metrics dict as a two-column Markdown table."""
    peak  = hw.get("peak_stack_bytes")
    total = hw.get("total_stack_bytes")
    t_ms  = hw.get("avg_inference_time_ms")

    rows = []
    if peak is not None:
        rows.append(["Peak internal Data RAM", f"{peak:,} B  ({peak/1024:.1f} KB)"])
    if total is not None:
        rows.append(["Total stack budget",     f"{total:,} B  ({total/1024:.1f} KB)"])
    if peak is not None and total is not None:
        headroom = hw.get("headroom_bytes", total - peak)
        rows.append(["Headroom",               f"{headroom:,} B  ({headroom/1024:.1f} KB)"])
    if t_ms is not None:
        fps = 1000 / t_ms if t_ms > 0 else 0
        rows.append(["Inference latency",      f"{t_ms:.0f} ms  (~{fps:.2f} FPS)"])

    SKIP = {"peak_stack_bytes", "total_stack_bytes", "headroom_bytes",
            "avg_inference_time_ms", "peak_stack_kb"}
    for k, v in hw.items():
        if k in SKIP:
            continue
        rows.append([k.replace("_", " ").title(), str(v)])

    return markdown_table(["Metric", "Value"], rows) if rows else ""


def section_hardware(results_root: Path) -> str:
    lines = ["## Hardware Profile (ESP32-S3)\n"]

    # Nano-U profiling data — written by scripts/stack_analyzer.py to results/nano_u/
    nano_u_hw = parse_hardware_profile(
        results_root / "nano_u" / "metrics.json",
        results_root / "nano_u" / "stack_log.txt",
    )
    if nano_u_hw:
        lines.append("### Nano-U\n")
        lines.append(_hw_table(nano_u_hw))
        lines.append("")
    else:
        lines.append("_Nano-U hardware profiling data not found. Run `python scripts/stack_analyzer.py` to generate it._\n")

    return "\n".join(lines)


def section_nas(results_root: Path) -> str:
    lines = ["## NAS Layer Redundancy (Final Epoch)\n"]

    found_any = False
    for dataset in DATASETS:
        csv_path = results_root / dataset / "nano_u" / "metrics.csv"
        rows_raw = parse_nas_csv(csv_path)
        if not rows_raw:
            continue

        last = rows_raw[-1]
        epoch = last.get("epoch", "?")

        # Group columns by layer (strip trailing _redundancy_score etc.)
        layers: dict[str, dict] = {}
        for col, val in last.items():
            if col == "epoch":
                continue
            # Column format: <layer>_<metric>
            for suffix in ("_redundancy_score", "_condition_number", "_rank", "_num_channels"):
                if col.endswith(suffix):
                    layer = col[: -len(suffix)]
                    layers.setdefault(layer, {})
                    layers[layer][suffix.lstrip("_")] = val
                    break

        if not layers:
            continue

        found_any = True
        lines.append(f"### {dataset.replace('_', ' ').title()} — Epoch {epoch}\n")

        table_rows = []
        for layer, metrics in layers.items():
            table_rows.append([
                layer,
                f"{float(metrics.get('redundancy_score', 0)):.4f}",
                f"{float(metrics.get('condition_number', 0)):.2f}",
                metrics.get("rank", "—"),
                metrics.get("num_channels", "—"),
            ])
        lines.append(markdown_table(
            ["Layer", "Redundancy Score", "Condition Number", "Rank", "Channels"],
            table_rows,
        ))
        lines.append("")

    if not found_any:
        lines.append("_No NAS metrics CSV found._\n")

    return "\n".join(lines)


def section_config_summary(results_root: Path) -> str:
    lines = ["## Training Configuration Summary\n"]

    for model in MODELS:
        lines.append(f"### {model}\n")
        for dataset in DATASETS:
            cfg_path = results_root / dataset / model / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = {}
            import re
            with cfg_path.open() as f:
                for line in f:
                    match = re.match(r'^(\w+):\s*(.*)$', line)
                    if match:
                        cfg[match.group(1)] = match.group(2).strip()

            keys = ["epochs", "batch_size", "learning_rate", "weight_decay",
                    "optimizer", "alpha", "temperature", "qat_enabled"]
            rows = []
            for k in keys:
                if k in cfg:
                    rows.append([k, str(cfg[k])])

            if rows:
                lines.append(f"**{dataset.replace('_', ' ').title()}**\n")
                lines.append(markdown_table(["Parameter", "Value"], rows))
                lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root    = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"
    output_path  = results_root / "summary.md"

    if not results_root.exists():
        print(f"Error: results/ not found at {results_root}", file=sys.stderr)
        sys.exit(1)

    sections = [
        f"# Nano-U — Results Summary\n",
        section_training(results_root),
        section_eval(results_root),
        section_hw_eval(results_root),
        section_quantization_gap(results_root),
        section_hardware(results_root),
        section_nas(results_root),
        section_config_summary(results_root),
    ]

    report = "\n".join(sections)
    output_path.write_text(report)
    print(f"Report written to {output_path}")

    # Also print to stdout so you can pipe it
    if "--print" in sys.argv:
        print("\n" + report)


if __name__ == "__main__":
    main()
