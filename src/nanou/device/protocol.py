"""Serial wire-format parsers.

Pure functions over strings and bytes: no port is opened here, nothing is
timed. The device scripts read lines; this module turns them into numbers.
"""

import base64
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_idx_us_line(line_str: str) -> tuple:
    """Parse 'PREFIX:{idx}:{us}us' (IMG_PREPROCESS / IMG_INFERENCE) -> (idx, us)."""
    parts = line_str.split(":")
    return int(parts[1]), int(parts[2].rstrip("us"))


def parse_stats_line(line_str: str) -> tuple:
    """Parse 'IMG_STATS:{idx}:{min},{max}' -> (idx, min, max)."""
    parts = line_str.split(":")
    idx = int(parts[1])
    min_v, max_v = parts[2].split(",")
    return idx, float(min_v), float(max_v)


def decode_hex_row(hex_str: str) -> np.ndarray:
    """Decode a row of little-endian f32 pixels from the firmware hex output."""
    return np.frombuffer(bytes.fromhex(hex_str), dtype="<f4")


def parse_log_output(filename):
    """Parse a monitor log into stack peaks, inference times and the total stack size.

    Args:
        filename: Path to a log written by `run_analysis_on_device`.

    Returns:
        Tuple of (peaks, inference times in ms, total stack in bytes).
    """
    peaks = []
    inf_times_ms = []
    total_stack = 0
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")

    with open(filename, errors="replace") as f:
        for line in f:
            line = ansi_escape.sub("", line).strip()

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

            # Accept both integer ms and float ms (e.g. "1234 ms" or "1.234 ms")
            if "Inference done in" in line:
                match = re.search(r"Inference done in ([0-9]+(?:\.[0-9]+)?)\s*ms", line)
                if match:
                    inf_times_ms.append(float(match.group(1)))

    avg_time = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else 0
    return peaks, total_stack, avg_time, inf_times_ms


BEGIN_RE = re.compile(
    r"^(?P<tag>\w+)_BEGIN idx=(?P<idx>\d+) w=(?P<w>\d+) h=(?P<h>\d+) "
    r"bytes=(?P<bytes>\d+) fmt=(?P<fmt>\w+)$"
)


def rgb565_to_rgb888(buf: bytes, w: int, h: int, swap: bool) -> np.ndarray:
    """Decode a big-endian (unless `swap`) RGB565 buffer to an (h, w, 3) uint8 array."""
    raw = np.frombuffer(buf, dtype=np.uint8).astype(np.uint16)
    if swap:
        px = raw[0::2] | (raw[1::2] << 8)
    else:
        px = (raw[0::2] << 8) | raw[1::2]
    px = px[: w * h].reshape(h, w)
    r5 = (px >> 11) & 0x1F
    g6 = (px >> 5) & 0x3F
    b5 = px & 0x1F
    r = (r5 << 3) | (r5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    b = (b5 << 3) | (b5 >> 2)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def save_block(meta: dict, b64_lines: list, out_dir: Path, upscale: int) -> Path:
    """Decode one completed BEGIN/END block and write it as a PNG."""
    data = base64.b64decode("".join(b64_lines))
    w, h, fmt = meta["w"], meta["h"], meta["fmt"]

    if fmt == "rgb565":
        arr = rgb565_to_rgb888(data, w, h, meta["swap"])
    elif fmt == "rgb888":
        arr = np.frombuffer(data, dtype=np.uint8)[: w * h * 3].reshape(h, w, 3)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")

    img = Image.fromarray(arr, "RGB")
    if upscale > 1:
        img = img.resize((w * upscale, h * upscale), Image.NEAREST)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame{meta['idx']:04d}_{meta['tag']}.png"
    img.save(path)
    return path
