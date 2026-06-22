#!/usr/bin/env python3
"""Decode and save the frames streamed by the firmware `capture` binary.

Companion to `firmware/src/bin/capture.rs`. That binary stops the live pipeline
short of inference and streams the intermediate images over serial so you can see
what the camera sees and validate each stage on its own:

  * RAW  — QQVGA 160x120 RGB565 straight from the OV2640 (sensor / framing /
           byte-order check)
  * DOWN — 60x80 RGB888 after the 2x2 box downscale (resize check; these are the
           exact pixels the INT8 quantizer turns into the model input)

Wire format (one block per stage per frame):

    <TAG>_BEGIN idx=<n> w=<w> h=<h> bytes=<len> fmt=<rgb565|rgb888>
    <base64 line>
    ...
    <TAG>_END idx=<n>

Each decoded block is saved as `frame<idx>_<tag>.png` under the output dir.

Usage:
    python scripts/capture_view.py                 # /dev/ttyACM0, ./capture_out
    python scripts/capture_view.py -n 5            # stop after 5 frames
    python scripts/capture_view.py -p /dev/ttyACM1 -o /tmp/cam --upscale 4

Note: this only reads the serial stream. Flash/run the binary separately, e.g.
    MODELS_DIR=../models/TinyAgri cargo run --release --bin capture
or point this script at the port while the device is already running.
"""
import argparse
import base64
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial
from PIL import Image

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", default="/dev/ttyACM0", help="serial port")
    ap.add_argument("-b", "--baud", type=int, default=115200, help="baud rate")
    ap.add_argument("-o", "--out", default="capture_out", help="output directory")
    ap.add_argument("-n", "--frames", type=int, default=0,
                    help="stop after N frames (0 = until Ctrl-C)")
    ap.add_argument("--upscale", type=int, default=1,
                    help="nearest-neighbour upscale factor for the saved PNGs")
    ap.add_argument("--swap-rgb565", action="store_true",
                    help="treat RAW frames as little-endian RGB565 (match the "
                         "firmware SWAP_RGB565_BYTES flag if colours look wrong)")
    ap.add_argument("--no-reset", action="store_true",
                    help="don't pulse RTS/DTR to reset the board into normal boot")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ser = serial.Serial(args.port, args.baud, timeout=0.2)

    if not args.no_reset:
        # Reset into normal boot: RTS=EN(reset), DTR=GPIO0(boot). Both deasserted = run.
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        time.sleep(0.05)

    print(f"Listening on {args.port} @ {args.baud}; saving to {out_dir}/  (Ctrl-C to stop)")

    meta = None          # current block metadata, or None when outside a block
    b64_lines = []
    frames_done = set()
    buf = b""

    try:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                m = BEGIN_RE.match(line)
                if m:
                    meta = {
                        "tag": m["tag"],
                        "idx": int(m["idx"]),
                        "w": int(m["w"]),
                        "h": int(m["h"]),
                        "fmt": m["fmt"],
                        "swap": args.swap_rgb565,
                    }
                    b64_lines = []
                    continue

                if meta is not None and line.startswith(f"{meta['tag']}_END"):
                    path = save_block(meta, b64_lines, out_dir, args.upscale)
                    print(f"  saved {path}")
                    meta = None
                    b64_lines = []
                    continue

                if meta is not None:
                    b64_lines.append(line)
                    continue

                # Outside a block: echo status lines (banner, FRAME_DONE, errors).
                print(line)
                fd = re.match(r"FRAME_DONE idx=(\d+)", line)
                if fd:
                    frames_done.add(int(fd.group(1)))
                    if args.frames and len(frames_done) >= args.frames:
                        print(f"Captured {len(frames_done)} frame(s); done.")
                        return
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
