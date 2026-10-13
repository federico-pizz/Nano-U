"""Capture RAW and downsampled frames from the board and save them as PNGs."""

import argparse
import re
import time
from pathlib import Path

import serial

from nanou.device.protocol import BEGIN_RE, save_block


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--port", default="/dev/ttyACM0", help="serial port")
    ap.add_argument("-b", "--baud", type=int, default=115200, help="baud rate")
    ap.add_argument("-o", "--out", default="capture_out", help="output directory")
    ap.add_argument(
        "-n", "--frames", type=int, default=0, help="stop after N frames (0 = until Ctrl-C)"
    )
    ap.add_argument(
        "--upscale", type=int, default=1, help="nearest-neighbour upscale factor for the saved PNGs"
    )
    ap.add_argument(
        "--swap-rgb565",
        action="store_true",
        help="treat RAW frames as little-endian RGB565 (match the "
        "firmware SWAP_RGB565_BYTES flag if colours look wrong)",
    )
    ap.add_argument(
        "--no-reset",
        action="store_true",
        help="don't pulse RTS/DTR to reset the board into normal boot",
    )
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

    meta = None  # current block metadata, or None when outside a block
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
